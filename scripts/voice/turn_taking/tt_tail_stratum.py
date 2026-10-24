#!/usr/bin/env python3
"""CS-45.1: tail-length / room-tone stratum, re-cut from the frozen CS-45 v2 pairs.  SYNTHESIZED, RANKING-ONLY.

Why: every v2 clip ends exactly 2.0 s (TRAIL) of DIGITAL silence after the user's last sample, so silence-gated and acoustic
arms fire on nearly all incomplete clips and any T_long >= 2.5 s arm is censored by the clip end (CS-47 addendum A). This
stratum re-cuts the CUT_IC / CUT_WAIT pairs with trailing regions of {0.3, 0.6, 1.0, 3.0, 4.0} s of ROOM TONE.

Recipe (a clip is a deterministic function of the v2 corpus row, the tail length and the pair id; nothing is stored but
the manifest, the PCM-SHA-256 and the generator constants):
  1. parent = tt_corpus.assemble_row(v2 row)                       (the ONE assembly function; v2 tail = TRAIL)
  2. keep samples [0, user_end + tail_samples); extend with zeros if the tail is longer than TRAIL
  3. ch0 (USER) outside the user segment [user_onset, user_end) := room tone bed (same bed across the pair, seeded by
     sha256(pair_id|tail|bed)); the speech segment itself is untouched, so the parent's user-segment ASR-B results
     (U1/U2/U3) remain valid and are BOUND by the segment's PCM-SHA-256 instead of re-run
  4. ch1 (ASSISTANT) is the parent's, untouched (an 'ended' assistant context has finished long before the user starts)

Room tone is SYNTHESIZED (integer Voss-McCartney pink noise scaled to ROOM_TONE_DBFS RMS), not a recording: it removes
the exact-zero cue and the 2.0 s length cue but is not natural ambience. The stratum therefore stays `synthesized`
and can only RANK or REJECT an arm; `advance_eligible_strata` stays ["human_recorded"].

CTX_BC pairs are out of scope: their tail is fixed by the continuing assistant's natural end (tt_corpus.ctx_tail), which is
longer than 0.3-1.0 s and not a free variable.

Natural-speech control: a declared-empty slot until CS-4 set 5 is recorded (see natural_control in tail_stratum.json).

  python3 tt_tail_stratum.py build  --v2 data/turn-taking-cs45-v1 --out data/turn-taking-cs45-1-tail-v1 [--audio-out DIR]
  python3 tt_tail_stratum.py verify --v2 data/turn-taking-cs45-v1 --dir data/turn-taking-cs45-1-tail-v1
  python3 tt_tail_stratum.py materialize --v2 ... --dir ... --out DIR
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

STRATUM_ID = "turn-taking-cs45-1-tail-v1"
INSTRUMENT_VERSION = "tt-tail-1"
TAILS_S = (0.3, 0.6, 1.0, 3.0, 4.0)
ROOM_TONE_DBFS = -55.0  # RMS of the synthesized bed (int16 full scale = 0 dBFS)
ROOM_TONE_TOL_DB = 1.5
FRAME_CEILING_DBFS = -40.0  # no 10 ms frame of the bed may reach speech-like level
ZERO_FRACTION_MAX = 0.01  # the tail must not be digital silence
FAMILIES = ("CUT_IC", "CUT_WAIT")
ADVANCE_ELIGIBLE = ["human_recorded"]
SR = A.SR


def tail_tag(tail_s: float) -> str:
    return f"t{int(round(tail_s * 1000)):04d}"


def tail_samples(tail_s: float) -> int:
    n = int(round(tail_s * SR))
    if abs(n / SR - tail_s) > 1e-9:
        raise ValueError(f"tail {tail_s} s is not a whole number of samples")
    return n


def bed_seed(pair_id: str, tail_s: float) -> int:
    return int.from_bytes(hashlib.sha256(f"{pair_id}|{tail_tag(tail_s)}|bed".encode()).digest()[:8], "big")


def room_tone(n: int, seed: int, rms_dbfs: float = ROOM_TONE_DBFS) -> np.ndarray:
    """Deterministic pink-ish room tone: integer Voss-McCartney rows (hold times 1, 2, 4, ... 128 samples) plus a white row,
    scaled to the target RMS and rounded to int16. n samples."""
    rng = np.random.default_rng(seed)
    rows = 9
    acc = np.zeros(n, dtype=np.int64)
    for k in range(rows):
        hold = 1 << k
        m = -(-n // hold)
        vals = rng.integers(-1000, 1001, size=m, dtype=np.int64)
        acc += np.repeat(vals, hold)[:n]
    acc += rng.integers(-1000, 1001, size=n, dtype=np.int64)
    x = acc.astype(np.float64)
    x -= x.mean()
    rms = float(np.sqrt((x ** 2).mean()))
    target = 32768.0 * 10 ** (rms_dbfs / 20.0)
    return np.clip(np.rint(x * (target / rms)), -32768, 32767).astype(np.int16)


def recut(parent: np.ndarray, lay: C.Layout, pair_id: str, tail_s: float) -> tuple[np.ndarray, dict]:
    """parent: v2 stereo clip [n,2] with layout lay. Returns (new stereo clip, layout dict)."""
    ts = tail_samples(tail_s)
    total = lay.user_end + ts
    out = np.zeros((total, 2), dtype=np.int16)
    m = min(total, len(parent))
    out[:m] = parent[:m]
    bed = room_tone(total, bed_seed(pair_id, tail_s))
    mask = np.ones(total, dtype=bool)
    mask[lay.user_onset : lay.user_end] = False
    out[mask, 0] = bed[mask]
    return out, {"user_onset": lay.user_onset, "user_len": lay.user_len, "user_end": lay.user_end, "total": total,
                 "assistant_start": lay.assistant_start, "assistant_end": lay.assistant_end, "tail": ts}


def level_dbfs(x: np.ndarray) -> float:
    return float(20 * np.log10(np.sqrt((x.astype(np.float64) ** 2).mean() + 1e-20) / 32768.0 + 1e-12))


def tail_gates(new: np.ndarray, lay: dict, parent: np.ndarray, parent_lay: C.Layout, tail_s: float) -> dict:
    """Deterministic gates on the ASSEMBLED array (no ASR). T3 is what lets the v2 ASR-B gate results carry over."""
    g = {}
    ts = tail_samples(tail_s)
    tail = new[lay["user_end"] :, 0]
    g["T1_tail_length"] = {"ok": bool(len(tail) == ts and lay["total"] == len(new) == lay["user_end"] + ts), "tail_samples": int(len(tail)), "expected": ts}
    zf = float((tail == 0).mean()) if len(tail) else 1.0
    lv = level_dbfs(tail) if len(tail) else -999.0
    nfr = len(tail) // 160
    fr = tail[: nfr * 160].astype(np.float64).reshape(nfr, 160) / 32768.0 if nfr else np.zeros((0, 160))
    fdb = 20 * np.log10(np.sqrt((fr ** 2).mean(axis=1) + 1e-20) + 1e-12) if nfr else np.array([-999.0])
    g["T2_room_tone"] = {"ok": bool(zf <= ZERO_FRACTION_MAX and abs(lv - ROOM_TONE_DBFS) <= ROOM_TONE_TOL_DB and float(fdb.max()) < FRAME_CEILING_DBFS),
                         "zero_fraction": round(zf, 5), "rms_dbfs": round(lv, 2), "max_frame_dbfs": round(float(fdb.max()), 2)}
    seg_new = new[lay["user_onset"] : lay["user_end"], 0]
    seg_par = parent[parent_lay.user_onset : parent_lay.user_end, 0]
    seg_new = new[lay["user_onset"] : lay["user_end"], 0]
    seg_par = parent[parent_lay.user_onset : parent_lay.user_end, 0]
    same_seg = bool(np.array_equal(seg_new, seg_par))
    m = min(len(new), len(parent))
    same_asst = bool(np.array_equal(new[:m, 1], parent[:m, 1]) and not np.any(new[m:, 1]))
    g["T3_bound_to_parent"] = {"ok": bool(same_seg and same_asst and lay["user_onset"] == parent_lay.user_onset and lay["user_end"] == parent_lay.user_end),
                               "user_segment_identical": same_seg, "assistant_channel_identical": same_asst}
    return g


def _read(path: Path):
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def _load_v2(v2: Path):
    recs = {r["recording_id"]: r for r in _read(v2 / "manifest" / "recordings.jsonl")}
    clips = _read(v2 / "manifest" / "clips.jsonl")
    pairs = {p["pair_id"]: p for p in _read(v2 / "manifest" / "pairs.jsonl")}
    qa = json.loads((v2 / "qa" / "qa_gates_report.json").read_text())
    return recs, clips, pairs, qa


def _sha_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def natural_control_status(v2: Path) -> dict:
    """The natural-speech control slot: filled only from human-recorded clips (CS-4 set 5) that exist in the v2 human slot."""
    hp = v2 / "manifest" / "human_recorded.jsonl"
    rows = _read(hp) if hp.exists() else []
    return {"slot": "natural_control", "source": "CS-4:set5 human-recorded clips (v2 manifest/human_recorded.jsonl)",
            "status": "filled" if rows else "declared-empty", "n": len(rows),
            "reason": ("" if rows else "CS-4 sets 2/5 are not recorded; manifest/human_recorded.jsonl has 0 rows. Nothing here is natural speech."),
            "human_recorded_jsonl_sha256": _sha_file(hp) if hp.exists() else None,
            "fill_procedure": "append rows to the v2 human_recorded.jsonl per HUMAN_SLOT.md, then rebuild this stratum: filled rows are carried unchanged (their own tails), tagged stratum=human_recorded"}


def build(v2: Path, out: Path, audio_out: Path | None = None, tails=TAILS_S, limit: int | None = None) -> dict:
    recs, clips, pairs, qa = _load_v2(v2)
    cache: dict[str, np.ndarray] = {}

    def rec(rid):
        if rid not in cache:
            cache[rid] = A.load_audio(v2, recs[rid]["path"])[0].samples[:, 0]
        return cache[rid]

    sel = [c for c in clips if c["family"] in FAMILIES]
    if limit:
        keep = sorted({c["pair_id"] for c in sel})[:limit]
        sel = [c for c in sel if c["pair_id"] in keep]
    rows, reports = [], {}
    by_pair: dict[str, list] = {}
    for c in sel:
        parent_q = qa["clips"].get(c["clip_id"])
        if not parent_q or not parent_q.get("pass") or parent_q["pcm_sha256"] != c["pcm"]["pcm_sha256"]:
            raise SystemExit(f"{c['clip_id']}: no passing v2 independent-gate result bound to its PCM-SHA-256")
        parent, play = C.assemble_row(c, rec(c["user"]["recording_id"]), rec(c["assistant"]["recording_id"]))
        if A.Pcm(parent, SR).pcm_sha256() != c["pcm"]["pcm_sha256"]:
            raise SystemExit(f"{c['clip_id']}: v2 parent PCM-SHA-256 mismatch")
        for ts_ in tails:
            new, lay = recut(parent, play, c["pair_id"], ts_)
            g = tail_gates(new, lay, parent, play, ts_)
            pcm = A.Pcm(new, SR)
            tag = tail_tag(ts_)
            cid = f"{c['clip_id']}.{tag}"
            row = {"clip_id": cid, "parent_clip_id": c["clip_id"], "parent_pcm_sha256": c["pcm"]["pcm_sha256"],
                   "pair_id": f"{c['pair_id']}.{tag}", "parent_pair_id": c["pair_id"], "pair_role": c["pair_role"],
                   "family": c["family"], "label": c["label"], "split": c["split"], "speaker_id": c["speaker_id"], "dialogue_id": c["dialogue_id"],
                   "stratum": "synthesized", "ranking_only": True, "tail_s": ts_, "tail_kind": "room_tone_synthesized",
                   "text": c["text"], "assistant": c["assistant"], "user": c["user"],
                   "gold": {**c["gold"], "clip_end_s": round(lay["total"] / SR, 4), "tail_s": ts_},
                   "layout": lay,
                   "room_tone": {"generator": "voss-mccartney-int-9rows+white", "seed": bed_seed(c["pair_id"], ts_), "rms_dbfs": ROOM_TONE_DBFS,
                                 "bed_regions": "ch0 outside [user_onset, user_end)", "numpy": np.__version__},
                   "pcm": {"sample_rate": SR, "channels": 2, "n_samples": int(pcm.n_samples), "pcm_sha256": pcm.pcm_sha256(),
                           "ch_sha256": [pcm.channel_sha256(0), pcm.channel_sha256(1)]}}
            qrow = {"clip_id": cid, "pcm_sha256": row["pcm"]["pcm_sha256"], "parent_clip_id": c["clip_id"],
                    "inherited_v2_gates": {"U1_U2_U3_A2": "pass", "bound_by": "user-segment + assistant-channel byte identity (T3)", "v2_pcm_sha256": c["pcm"]["pcm_sha256"]},
                    "gates": g, "pass": all(v["ok"] for v in g.values())}
            reports[cid] = qrow
            rows.append(row)
            by_pair.setdefault(row["pair_id"], []).append(row)
            if audio_out is not None:
                A.write_wav(audio_out / f"{cid}.wav", pcm)
    # pair-level gate T4: members share a tail and the bed agrees where both beds cover the same clip-clock samples
    prow = []
    for pid, members in sorted(by_pair.items()):
        ok = len(members) == 2 and members[0]["tail_s"] == members[1]["tail_s"] and {m["pair_role"] for m in members} == {"a", "b"}
        prow.append({"pair_id": pid, "parent_pair_id": members[0]["parent_pair_id"], "family": members[0]["family"], "tail_s": members[0]["tail_s"],
                     "clips": [m["clip_id"] for m in sorted(members, key=lambda m: m["pair_role"])], "labels": {m["pair_role"]: m["label"] for m in members},
                     "split": members[0]["split"], "stratum": "synthesized", "ranking_only": True, "T4_pair_complete": ok})
        for m in members:
            reports[m["clip_id"]]["gates"]["T4_pair_complete"] = {"ok": ok}
            reports[m["clip_id"]]["pass"] = reports[m["clip_id"]]["pass"] and ok
    bad = sorted(k for k, v in reports.items() if not v["pass"])
    out.mkdir(parents=True, exist_ok=True)
    (out / "manifest").mkdir(exist_ok=True)
    (out / "qa").mkdir(exist_ok=True)
    (out / "manifest" / "clips.jsonl").write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))
    (out / "manifest" / "pairs.jsonl").write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in prow))
    (out / "manifest" / "natural_control.jsonl").write_text("")  # declared-empty unless v2 human_recorded.jsonl has rows
    qrep = {"gates_version": "tt-tail-gates-1", "inherits": "tt-qa-gates-2 (ASR-B faster-whisper large-v3 int8 CPU) on the byte-identical user segment / assistant channel",
            "n_clips": len(reports), "defective_clips": len(bad), "all_pass": not bad, "clips": reports}
    (out / "qa" / "qa_tail_gates_report.json").write_text(json.dumps(qrep, indent=1, sort_keys=True))
    meta = {"stratum_id": STRATUM_ID, "instrument_version": INSTRUMENT_VERSION, "parent_corpus": "turn-taking-cs45-v1",
            "parent_corpus_json_sha256": _sha_file(v2 / "corpus.json"), "stratum": "synthesized", "ranking_only": True,
            "advance_eligible_strata": ADVANCE_ELIGIBLE, "tails_s": list(tails), "families": list(FAMILIES), "excluded_families": {"CTX_BC": "tail fixed by the continuing assistant's natural end"},
            "room_tone": {"kind": "synthesized (not a recording)", "rms_dbfs": ROOM_TONE_DBFS, "tol_db": ROOM_TONE_TOL_DB, "frame_ceiling_dbfs": FRAME_CEILING_DBFS},
            "counts": {"parent_pairs": len({c["pair_id"] for c in sel}), "pairs": len(prow), "clips": len(rows), "defective_clips": len(bad)},
            "natural_control": natural_control_status(v2),
            "files": {p: _sha_file(out / p) for p in ("manifest/clips.jsonl", "manifest/pairs.jsonl", "manifest/natural_control.jsonl", "qa/qa_tail_gates_report.json")}}
    (out / "tail_stratum.json").write_text(json.dumps(meta, indent=1, sort_keys=True))
    return meta


def verify(v2: Path, d: Path) -> list[str]:
    """Return a list of violations (empty = stratum is valid): re-assembles every clip from the v2 corpus and checks the hash."""
    bad: list[str] = []
    meta = json.loads((d / "tail_stratum.json").read_text())
    if meta.get("advance_eligible_strata") != ["human_recorded"]:
        bad.append("advance_eligible_strata widened or changed")
    if meta.get("stratum") != "synthesized" or meta.get("ranking_only") is not True:
        bad.append("stratum is not marked synthesized / ranking-only")
    for p, h in meta["files"].items():
        if _sha_file(d / p) != h:
            bad.append(f"file digest mismatch: {p}")
    if meta["parent_corpus_json_sha256"] != _sha_file(v2 / "corpus.json"):
        bad.append("parent corpus.json digest changed")
    nc = natural_control_status(v2)
    nrows = _read(d / "manifest" / "natural_control.jsonl")
    if nc["status"] == "declared-empty" and (nrows or meta["natural_control"]["status"] != "declared-empty"):
        bad.append("natural control claimed filled while the human-recorded slot is empty")
    if nc["status"] == "filled" and meta["natural_control"]["n"] != nc["n"]:
        bad.append("natural control stale versus the human-recorded slot")
    recs, clips, _pairs, qa = _load_v2(v2)
    byid = {c["clip_id"]: c for c in clips}
    cache: dict[str, np.ndarray] = {}

    def rec(rid):
        if rid not in cache:
            cache[rid] = A.load_audio(v2, recs[rid]["path"])[0].samples[:, 0]
        return cache[rid]

    qrep = json.loads((d / "qa" / "qa_tail_gates_report.json").read_text())
    for r in _read(d / "manifest" / "clips.jsonl"):
        if r.get("stratum") != "synthesized" or not r.get("ranking_only"):
            bad.append(f"{r['clip_id']}: not marked synthesized/ranking-only")
        pc = byid.get(r["parent_clip_id"])
        if pc is None or pc["pcm"]["pcm_sha256"] != r["parent_pcm_sha256"]:
            bad.append(f"{r['clip_id']}: parent missing or changed")
            continue
        pq = qa["clips"].get(pc["clip_id"])
        if not pq or not pq["pass"] or pq["pcm_sha256"] != pc["pcm"]["pcm_sha256"]:
            bad.append(f"{r['clip_id']}: parent has no passing v2 gate result")
        parent, play = C.assemble_row(pc, rec(pc["user"]["recording_id"]), rec(pc["assistant"]["recording_id"]))
        new, lay = recut(parent, play, pc["pair_id"], r["tail_s"])
        if A.Pcm(new, SR).pcm_sha256() != r["pcm"]["pcm_sha256"]:
            bad.append(f"{r['clip_id']}: PCM-SHA-256 mismatch on re-assembly")
        q = qrep["clips"].get(r["clip_id"])
        if not q or not q["pass"] or q["pcm_sha256"] != r["pcm"]["pcm_sha256"]:
            bad.append(f"{r['clip_id']}: tail gate result missing, failing or not bound to the PCM-SHA-256")
        g = tail_gates(new, lay, parent, play, r["tail_s"])
        if not all(v["ok"] for v in g.values()):
            bad.append(f"{r['clip_id']}: tail gates fail on re-assembly")
    return bad


def materialize(v2: Path, d: Path, out: Path, ids: set | None = None) -> int:
    recs, clips, _p, _q = _load_v2(v2)
    byid = {c["clip_id"]: c for c in clips}
    cache: dict[str, np.ndarray] = {}

    def rec(rid):
        if rid not in cache:
            cache[rid] = A.load_audio(v2, recs[rid]["path"])[0].samples[:, 0]
        return cache[rid]

    n = 0
    for r in _read(d / "manifest" / "clips.jsonl"):
        if ids and r["clip_id"] not in ids:
            continue
        pc = byid[r["parent_clip_id"]]
        parent, play = C.assemble_row(pc, rec(pc["user"]["recording_id"]), rec(pc["assistant"]["recording_id"]))
        new, _ = recut(parent, play, pc["pair_id"], r["tail_s"])
        pcm = A.Pcm(new, SR)
        if pcm.pcm_sha256() != r["pcm"]["pcm_sha256"]:
            raise SystemExit(f"{r['clip_id']}: PCM-SHA-256 mismatch")
        A.write_wav(out / f"{r['clip_id']}.wav", pcm)
        n += 1
    return n


def main() -> int:
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    b = sp.add_parser("build")
    b.add_argument("--v2", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--audio-out", default=None)
    b.add_argument("--limit", type=int, default=None)
    v = sp.add_parser("verify")
    v.add_argument("--v2", required=True)
    v.add_argument("--dir", required=True)
    m = sp.add_parser("materialize")
    m.add_argument("--v2", required=True)
    m.add_argument("--dir", required=True)
    m.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.cmd == "build":
        meta = build(Path(a.v2), Path(a.out), Path(a.audio_out) if a.audio_out else None, limit=a.limit)
        print(json.dumps(meta["counts"]))
        return 0 if meta["counts"]["defective_clips"] == 0 else 1
    if a.cmd == "verify":
        bad = verify(Path(a.v2), Path(a.dir))
        print("VERDICT", "PASS" if not bad else "FAIL", *bad[:20], sep="\n" if bad else " ")
        return 1 if bad else 0
    print(f"materialised {materialize(Path(a.v2), Path(a.dir), Path(a.out))} clips, all PCM-SHA-256 verified")
    return 0


if __name__ == "__main__":
    sys.exit(main())
