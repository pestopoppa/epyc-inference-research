#!/usr/bin/env python3
"""CS-45 independent QA gates (written after the 2026-10-20 human review found 12 of 20 sampled pairs defective).

The corpus is BUILT with ASR-A (faster-whisper large-v3-turbo word timestamps, tt_align_fw.py) plus an energy rule and
SCORED with the pinned whisper.cpp large-v3-turbo. These gates use ASR-B = faster-whisper **large-v3** (different model,
different decoder depth, different weights from both) on the FINAL clips, plus energy checks, so they are not circular:

  U1  user-channel word sequence equals the expected text EXACTLY (prefix for a cut member b, full text otherwise)
  U2  no partial word at the cut (cut members only): ASR-B on the SUFFIX of the recording (everything after the cut)
      equals the expected suffix exactly (a word cut in half cannot produce its whole first word), the prefix check U1
      rejects an extra fragment, and the cut-point level is >= LEVEL_DB_BELOW_P95 dB below the recording's p95 level.
      (ASR-B word TIMESTAMPS are recorded as a diagnostic only: cross-attention timestamps are good to ~0.1 s, too coarse
      to gate a 10 ms cut.)
  U3  user channel non-empty: >= 0.2 s of frames at speech level (recording p95 - 24 dB) AND ASR-B returns words
  A1  (assistant-continuing members) the assistant channel is the COMPLETE monologue: ASR-B transcript equals the
      assistant text exactly, it is still speaking >= N_CONT s after the user's backchannel onset, and the clip ends
      only after it finished
  A2  (all members) the assistant channel transcript equals the assistant text exactly (once per unique
      recording+mode combination)

Runs under the faster-whisper interpreter (numpy only; no torch, no scipy):
  /mnt/raid0/llm/pace-env/bin/python -I tt_qa_gates.py --corpus data/turn-taking-cs45-v1 --out report.json \
      [--which main,reserve] [--threads 16] [--limit N]
Run it inside a region-lock build claim (it is CPU work). No server, no GPU, no network.
"""
from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tt_audio as A  # noqa: E402
import tt_corpus as C  # noqa: E402

FW_LARGE_V3 = "/mnt/raid0/llm/cache/huggingface/hub/models--Systran--faster-whisper-large-v3/snapshots/edaa852ec7e145841d8ffdb056a99866b5f0a478"
TOL_S = 0.060
LEVEL_DB_BELOW_P95 = 12.0   # cut point must sit at least this far below the recording's p95 10-ms RMS (energy, not ASR)
N_CONT_S = 3.0              # assistant must still be speaking this long after the backchannel onset
MIN_SPEECH_S = 0.25

_ONES = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split()
_TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()


def _num_words(n: int) -> str:
    if n < 20:
        return _ONES[n]
    if n < 100:
        return _TENS[n // 10] + ("" if n % 10 == 0 else " " + _ONES[n % 10])
    return str(n)


def norm_words(text: str) -> list[str]:
    t = text.lower().replace("-", " ").replace("%", " percent ")
    t = re.sub(r"(\d+):(\d+)", lambda m: f"{_num_words(int(m.group(1)))} {_num_words(int(m.group(2)))}", t)
    t = re.sub(r"\d+", lambda m: _num_words(int(m.group(0))), t)
    return re.sub(r"[^a-z' ]+", " ", t).split()


class AsrB:
    def __init__(self, threads: int, cache: str | None = None, extra_caches: list[str] | None = None, cache_only: bool = False):
        self.cache_only = cache_only
        if not cache_only:
            from faster_whisper import WhisperModel

            self.m = WhisperModel(FW_LARGE_V3, device="cpu", compute_type="int8", cpu_threads=threads)
        self.n_calls = 0
        self.cache_path = Path(cache) if cache else None
        self.cache: dict[str, dict] = {}
        for p in [cache] + list(extra_caches or []):
            if p and Path(p).exists():
                for l in Path(p).read_text().splitlines():
                    if l.strip():
                        d = json.loads(l)
                        self.cache[d["k"]] = d["v"]

    def words(self, x_i16: np.ndarray, ctx: dict | None = None) -> dict:
        """x_i16: mono int16 @16 kHz. 0.3 s of digital-zero lead-in only (a zero TAIL makes Whisper hallucinate
        'Thank you for watching'); the lead-in is removed from the times. ctx is ignored (used by test stubs)."""
        import hashlib

        key = hashlib.sha256(np.ascontiguousarray(x_i16, dtype="<i2").tobytes()).hexdigest()
        if key in self.cache:
            return self.cache[key]
        if self.cache_only:
            return {"words": [], "text": "", "missing": True}
        pad = int(0.3 * A.SR)
        x = np.concatenate([np.zeros(pad, np.float32), x_i16.astype(np.float32) / 32768.0])
        segs, _ = self.m.transcribe(x, language="en", word_timestamps=True, vad_filter=False, beam_size=5, temperature=0.0,
                                    condition_on_previous_text=False, without_timestamps=False)
        ws = []
        for s in segs:
            for w in s.words or []:
                ws.append({"w": w.word.strip(), "t0": round(w.start - 0.3, 3), "t1": round(w.end - 0.3, 3), "p": round(w.probability, 3)})
        self.n_calls += 1
        res = {"words": ws, "text": " ".join(w["w"] for w in ws)}
        self.cache[key] = res
        if self.cache_path and not self.cache_only:
            with open(self.cache_path, "a") as f:
                f.write(json.dumps({"k": key, "v": res}) + "\n")
        return res


def covers(exp: list[str], got: list[str], max_tail: int = 4) -> bool:
    """True iff every expected word appears in order in got and the only extra words are a short TRAILING run
    (Whisper habitually appends 'thank you for watching' or repeats the last word after an utterance that ends
    naturally). A truncated or garbled assistant context has missing words and fails."""
    sm = difflib.SequenceMatcher(a=exp, b=got, autojunk=False)
    ops = [o for o in sm.get_opcodes() if o[0] != "equal"]
    for tag, i1, i2, j1, j2 in ops:
        if tag != "insert" or j2 != len(got) or (j2 - j1) > max_tail:
            return False
    return True


def level_db(x: np.ndarray, center: int, half: int = 80) -> float:
    s = x[max(0, center - half) : center + half].astype(np.float64) / 32768.0
    return float(20 * np.log10(np.sqrt((s ** 2).mean() + 1e-20) + 1e-12))


def p95_db(x: np.ndarray) -> float:
    n = len(x) // 160
    fr = x[: n * 160].astype(np.float64).reshape(n, 160) / 32768.0
    db = 20 * np.log10(np.sqrt((fr ** 2).mean(axis=1) + 1e-20) + 1e-12)
    v = db[db > -70]
    return float(np.percentile(v, 95)) if len(v) else -20.0


def map_index(intended: list[str], hyp: list[str]) -> dict[int, int]:
    sm = difflib.SequenceMatcher(a=intended, b=hyp, autojunk=False)
    m = {}
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for k in range(i2 - i1):
                m[i1 + k] = j1 + k
    return m


def load_set(D: Path, which: str):
    base = D / "manifest" / ("reserve" if which == "reserve" else "")
    recs = {r["recording_id"]: r for r in map(json.loads, (D / "manifest" / "recordings.jsonl").read_text().splitlines())}
    if which == "reserve":
        recs.update({r["recording_id"]: r for r in map(json.loads, (base / "recordings.jsonl").read_text().splitlines())})
    clips = [json.loads(x) for x in (base / "clips.jsonl").read_text().splitlines()]
    pairs = [json.loads(x) for x in (base / "pairs.jsonl").read_text().splitlines()]
    return recs, clips, pairs


def run(D: Path, which: str, threads: int, limit: int | None, asr: AsrB | None = None, progress=print, reuse: dict | None = None, pair_filter: str | None = None) -> dict:
    recs, clips, pairs = load_set(D, which)
    if limit:
        keep = {p["pair_id"] for p in pairs[:limit]}
        pairs = [p for p in pairs if p["pair_id"] in keep]
        clips = [c for c in clips if c["pair_id"] in keep]
    if pair_filter:
        import re as _re

        keep = {p["pair_id"] for p in pairs if _re.search(pair_filter, p["pair_id"])}
        pairs = [p for p in pairs if p["pair_id"] in keep]
        clips = [c for c in clips if c["pair_id"] in keep]
    asr = asr or AsrB(threads)
    pcm: dict[str, np.ndarray] = {}

    def rec(rid):
        if rid not in pcm:
            pcm[rid] = A.load_audio(D, recs[rid]["path"])[0].samples[:, 0]
        return pcm[rid]

    # main-set assistant recordings are referenced by reserve clips too
    if which == "reserve":
        mrecs = {r["recording_id"]: r for r in map(json.loads, (D / "manifest" / "recordings.jsonl").read_text().splitlines())}
        recs.update(mrecs)
    asm = {}
    t0 = time.time()
    out_clips = {}
    asst_cache: dict[tuple, dict] = {}
    for i, c in enumerate(clips):
        arr, lay = C.assemble_clip(rec(c["user"]["recording_id"]), tuple(c["user"]["trim"]), c["user"]["cut"], rec(c["assistant"]["recording_id"]), c["assistant"]["mode"],
                                   **({"tail": c["layout"].get("tail")} if "tail" in c["layout"] else {}))
        asm[c["clip_id"]] = (arr, lay)
    reused = {}
    if reuse:
        by_p: dict[str, list] = {}
        for c in clips:
            by_p.setdefault(c["pair_id"], []).append(c)
        known = {pid for pid, cs in by_p.items() if all(A.Pcm(asm[c["clip_id"]][0], A.SR).pcm_sha256() in reuse for c in cs)}
        for c in clips:
            if c["pair_id"] in known:
                e = dict(reuse[A.Pcm(asm[c["clip_id"]][0], A.SR).pcm_sha256()])
                e["clip_id"] = c["clip_id"]
                reused[c["clip_id"]] = e
        clips = [c for c in clips if c["pair_id"] not in known]
        progress(f"  reusing {len(reused)} clip results (identical PCM-SHA-256), gating {len(clips)} new clips")
    unevaluated = {}
    if getattr(asr, "cache_only", False):
        import hashlib

        def k(x):
            return hashlib.sha256(np.ascontiguousarray(x, dtype="<i2").tobytes()).hexdigest()

        by_p2: dict[str, list] = {}
        for c in clips:
            by_p2.setdefault(c["pair_id"], []).append(c)
        bad_p = set()
        for pid_, cs in by_p2.items():
            for c in cs:
                arr, lay = asm[c["clip_id"]]
                need = [k(arr[lay.user_onset : lay.user_end, 0])]
                if c["family"] in ("CUT_IC", "CUT_WAIT") and c["user"]["cut"] is not None:
                    rc_ = rec(c["user"]["recording_id"])
                    t0_, t1_ = c["user"]["trim"]
                    suf_ = rc_[t0_ + c["user"]["cut"] : t1_].copy()
                    nf_ = min(80, len(suf_))
                    suf_[:nf_] = (suf_[:nf_].astype(np.float64) * np.linspace(0.0, 1.0, nf_)).astype(np.int16)
                    need.append(k(suf_))
                need.append(k(arr[lay.assistant_start : lay.assistant_end, 1]))
                if any(x not in asr.cache for x in need):
                    bad_p.add(pid_)
        for c in clips:
            if c["pair_id"] in bad_p:
                unevaluated[c["clip_id"]] = {"clip_id": c["clip_id"], "pair_id": c["pair_id"], "family": c["family"], "label": c["label"], "split": c["split"],
                                             "pcm_sha256": A.Pcm(asm[c["clip_id"]][0], A.SR).pcm_sha256(), "gates": {}, "pass": False, "unevaluated": True}
        clips = [c for c in clips if c["pair_id"] not in bad_p]
        progress(f"  cache-only: {len(unevaluated)} clips have uncached transcripts and are reported as unevaluated (not passing)")
    ci = 0
    for c in clips:
        arr, lay = asm[c["clip_id"]]
        u = arr[:, 0]
        seg = u[lay.user_onset : lay.user_end]
        a = asr.words(seg, {"kind": "user", "clip": c})
        r = {"clip_id": c["clip_id"], "pair_id": c["pair_id"], "family": c["family"], "label": c["label"], "split": c["split"],
             "pcm_sha256": A.Pcm(arr, A.SR).pcm_sha256(), "asr_user": a, "gates": {}}
        r["_seg"] = None
        out_clips[c["clip_id"]] = r
        ci += 1
        if ci % 20 == 0:
            progress(f"  user ASR {ci}/{len(clips)} ({time.time() - t0:.0f}s)")
    # ---- gate evaluation (needs a's full-recording alignment for b's boundary test)
    by_pair: dict[str, dict] = {}
    for c in clips:
        by_pair.setdefault(c["pair_id"], {})[c["pair_role"]] = c
    for pid, pr in by_pair.items():
        for role, c in pr.items():
            r = out_clips[c["clip_id"]]
            arr, lay = asm[c["clip_id"]]
            u = arr[:, 0]
            gold_text = c["text"]["user_spoken"]
            exp = norm_words(gold_text)
            hyp_raw = [w["w"] for w in r["asr_user"]["words"]]
            hyp = norm_words(" ".join(hyp_raw))
            g = r["gates"]
            fam = c["family"]
            # U3 non-empty
            seg = u[lay.user_onset : lay.user_end]
            thr = p95_db(rec(c["user"]["recording_id"])) - 20.0  # same speech-relative rule as the trim (a -42 dB noise floor is not speech)
            n = len(seg) // 160
            fr = seg[: n * 160].astype(np.float64).reshape(n, 160) / 32768.0
            db = 20 * np.log10(np.sqrt((fr ** 2).mean(axis=1) + 1e-20) + 1e-12)
            speech_s = float((db >= thr).sum() * 0.01)
            max_p = max([w_["p"] for w_ in r["asr_user"]["words"]] or [0.0])
            g["U3_nonempty"] = {"ok": speech_s >= MIN_SPEECH_S and len(hyp) > 0 and max_p >= 0.3, "speech_s": round(speech_s, 2), "asr_words": len(hyp), "max_word_prob": max_p}
            # U1 word sequence
            if fam == "CTX_BC":
                g["U1_words"] = {"ok": len(hyp) > 0, "expected": "(acknowledgement: non-empty only)", "got": " ".join(hyp)}
            else:
                g["U1_words"] = {"ok": hyp == exp, "expected": " ".join(exp), "got": " ".join(hyp)}
            # U2 no partial word at the cut (cut members only): suffix transcription + energy
            if fam in ("CUT_IC", "CUT_WAIT") and c["user"]["cut"] is not None:
                rc = rec(c["user"]["recording_id"])
                t0_, t1_ = c["user"]["trim"]
                cut_abs = t0_ + c["user"]["cut"]
                suf = rc[cut_abs:t1_].copy()
                nfade = min(80, len(suf))
                suf[:nfade] = (suf[:nfade].astype(np.float64) * np.linspace(0.0, 1.0, nfade)).astype(np.int16)  # 5 ms fade-in
                sw = asr.words(suf, {"kind": "suffix", "clip": c})
                nfull = norm_words(pr["a"]["text"]["user_full"])
                exp_suf = nfull[len(exp):]
                got_suf = norm_words(" ".join(w["w"] for w in sw["words"]))
                lvl = level_db(rc, cut_abs)
                p95 = p95_db(rc)
                ok_e = lvl <= p95 - LEVEL_DB_BELOW_P95
                diag = {}
                full = out_clips[pr["a"]["clip_id"]]["asr_user"]["words"]
                flat, idx = [], []
                for k, w_ in enumerate(full):
                    for t in norm_words(w_["w"]):
                        flat.append(t)
                        idx.append(k)
                mm = map_index(nfull, flat)
                if (len(exp) - 1) in mm and len(exp) in mm:
                    pw, nw = full[idx[mm[len(exp) - 1]]], full[idx[mm[len(exp)]]]
                    diag = {"asr_b_prev_end": pw["t1"], "asr_b_next_start": nw["t0"], "cut_t": round(c["layout"]["user_len"] / A.SR, 3)}
                g["U2_boundary"] = {"ok": bool(got_suf == exp_suf and ok_e), "suffix_expected": " ".join(exp_suf), "suffix_got": " ".join(got_suf),
                                    "level_db": round(lvl, 1), "p95_db": round(p95, 1), "energy_ok": bool(ok_e), "suffix_ok": bool(got_suf == exp_suf), **diag}
            # assistant gates
            if c["assistant"]["mode"] == "continuing":
                asst = rec(c["assistant"]["recording_id"])
                a_ch = arr[lay.assistant_start : lay.assistant_end, 1]
                aw = asr.words(a_ch, {"kind": "assistant", "clip": c})
                exp_a = norm_words(c["text"]["assistant"])
                got_a = norm_words(" ".join(w["w"] for w in aw["words"]))
                onset_s = lay.user_onset / A.SR
                last_speech = None
                n2 = len(arr) // 160
                f2 = arr[: n2 * 160, 1].astype(np.float64).reshape(n2, 160) / 32768.0
                d2 = 20 * np.log10(np.sqrt((f2 ** 2).mean(axis=1) + 1e-20) + 1e-12)
                idxs = np.where(d2 >= -45.0)[0]
                last_speech = float((idxs[-1] + 1) * 0.01) if len(idxs) else 0.0
                clip_end = len(arr) / A.SR
                cont_s = last_speech - onset_s
                complete = covers(exp_a, got_a)
                ends_clean = (clip_end - last_speech) >= 0.3
                g["A1_assistant_continues"] = {"ok": bool(complete and cont_s >= N_CONT_S and ends_clean), "assistant_complete": bool(complete), "continues_after_onset_s": round(cont_s, 2),
                                               "clip_ends_after_assistant_s": round(clip_end - last_speech, 2), "asr_assistant": " ".join(got_a)[:160], "expected": " ".join(exp_a)[:160]}
            else:
                key = (c["assistant"]["recording_id"],)
                if key not in asst_cache:
                    a_ch = arr[lay.assistant_start : lay.assistant_end, 1]
                    aw = asr.words(a_ch, {"kind": "assistant", "clip": c})
                    asst_cache[key] = {"got": norm_words(" ".join(w["w"] for w in aw["words"])), "exp": norm_words(c["text"]["assistant"])}
                v = asst_cache[key]
                g["A2_assistant_text"] = {"ok": covers(v["exp"], v["got"]), "got": " ".join(v["got"]), "expected": " ".join(v["exp"])}
            r["pass"] = all(x["ok"] for x in g.values())

            r.pop("_seg", None)
    out_clips.update(reused)
    out_clips.update(unevaluated)
    # ---- classification
    defects = {"U1_words": 0, "U2_boundary": 0, "U3_nonempty": 0, "A1_assistant_continues": 0, "A2_assistant_text": 0}
    bad_clips, bad_pairs = [], set()
    for cid, r in out_clips.items():
        for gname, gv in r["gates"].items():
            if not gv["ok"]:
                defects[gname] += 1
        if not r["pass"]:
            bad_clips.append(cid)
            bad_pairs.add(r["pair_id"])
    fam_counts = {}
    for p in pairs:
        f = fam_counts.setdefault(p["family"], {"pairs": 0, "defective_pairs": 0})
        f["pairs"] += 1
        f["defective_pairs"] += p["pair_id"] in bad_pairs
    return {"gates_version": "tt-qa-gates-2", "which": which, "asr_b": "faster-whisper large-v3 int8 CPU", "n_clips": len(out_clips), "n_pairs": len(pairs),
            "defect_counts_by_gate": defects, "defective_clips": len(bad_clips), "defective_pairs": len(bad_pairs), "by_family": fam_counts,
            "clips": out_clips}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--which", default="main,reserve")
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--asr-cache", default=None, help="jsonl cache of ASR-B transcripts keyed by audio sha256 (makes a crashed run cheap to repeat)")
    ap.add_argument("--asr-cache-extra", default="", help="comma list of extra read-only transcript caches")
    ap.add_argument("--cache-only", action="store_true", help="never run the model: a clip with an uncached transcript is reported as unevaluated (not passing)")
    ap.add_argument("--pair-filter", default=None, help="regex on candidate pair ids: gate only those pairs")
    ap.add_argument("--reuse", default=None, help="previous report: clips with an identical PCM-SHA-256 are not re-gated")
    a = ap.parse_args()
    D = Path(a.corpus)
    asr = AsrB(a.threads, a.asr_cache, [x for x in a.asr_cache_extra.split(",") if x], a.cache_only)
    rep = {}
    for w in a.which.split(","):
        print(f"== {w}", flush=True)
        reuse = None
        if a.reuse and Path(a.reuse).exists():
            reuse = {}
            for part in json.loads(Path(a.reuse).read_text()).values():
                for cid, e in part["clips"].items():
                    reuse[e["pcm_sha256"]] = e
        rep[w] = run(D, w, a.threads, a.limit, asr, reuse=reuse, pair_filter=a.pair_filter)
        print(json.dumps({k: v for k, v in rep[w].items() if k != "clips"}), flush=True)
    Path(a.out).write_text(json.dumps(rep, indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o)))
    ok = all(r["defective_pairs"] == 0 for r in rep.values())
    print("VERDICT", "PASS" if ok else "DEFECTS")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
