#!/usr/bin/env python3
"""CS-48 scorers and run-integrity gates. No cloud judge, no service, no torch.

score_run(run_dir) -> one JSON-serialisable dict; ``canonical_bytes`` makes it byte-reproducible (the double-replay
check). Sections:
  integrity   run refused unless hashes, lengths, channels, clock agreement and cancel-hook order all hold, for BOTH
              conditions (overlap: input/output; clean: clean_input/clean_output, the FDB v1.5 non-overlap baseline)
  fdb_timing  FDB v1.5 get_timing (pinned Silero only) under BOTH merge-gap readings (code 0.6/0.5, comment 0.3/0.05)
  eventlog    OUR deterministic t_stop / t_resp / chain latency / Respond-Resume labels (NON-comparable to FDB GPT-4o labels)
  humdial     first-response delay and timing rejection from the event log, cross-checked on Silero(output.wav)
  fdb_v1      pause_handling / smooth_turn_taking via whisper.cpp word timestamps (only when an ASR provider is given)
"""
from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import numpy as np

import bi_audio as A
import bi_fdb
import bi_log
import bi_vad
from bi_session import ns_to_sample, sample_to_ns

TOL_VAD_S = 0.20  # declared: Silero segment edges vs ground-truth sample times (32 ms window + 30 ms pad + onset lag)
CLOCK_TOL_SAMPLES = 160  # 10 ms: event-log playback_start vs first non-zero clean_output sample

MEASUREMENT_ERA = [
    {"id": "ME-CS48-1", "change": "FDB get_timing.py runs on the CS-45 patched copy (ME-FDB-1/2: no torch/torchaudio/torch.hub; "
     "pinned ONNX Silero); merge gaps passed explicitly per reading instead of editing module constants"},
    {"id": "ME-CS48-2", "change": "FDB v1 word timestamps from pinned whisper.cpp (CS-45 ME-FDB-4) replace CrisperWhisper/NeMo; "
     "FDB eval_pause_handling.py / eval_smooth_turn_taking.py run UNMODIFIED"},
    {"id": "ME-CS48-3", "change": "t_stop/t_resp Respond/Resume labels are event-log deterministic, NOT GPT-4o labels; "
     "no gpt-4-turbo interruption rating; values are not comparable to published FDB numbers"},
]


class RunRefused(ValueError):
    """Run directory fails an integrity gate; no score is produced."""


def canonical_bytes(d: dict) -> bytes:
    return (json.dumps(d, sort_keys=True, indent=1, ensure_ascii=True) + "\n").encode()


# ------------------------------------------------------------------ integrity
def load_run(run_dir: Path) -> dict:
    rd = Path(run_dir)
    try:
        run = json.loads((rd / "run.json").read_text())
    except FileNotFoundError as e:
        raise RunRefused("run.json missing") from e
    conds = {}
    for cond, (fin, fout, fev) in run["conditions"].items():
        chans = {}
        for f in (fin, fout):
            if not (rd / f).exists():
                raise RunRefused(f"CHANNEL_MISSING: {f} absent")
        if not (rd / fev).exists():
            raise RunRefused(f"EVENT_LOG: {fev} absent")
        for f in (fin, fout, fev):
            if A.file_sha256(rd / f) != run["files"][f]:
                raise RunRefused(f"HASH_MISMATCH: {f} differs from run.json (altered after recording)")
        for key, f in (("input", fin), ("output", fout)):
            x, sr = A.read_wav(rd / f)
            if sr != run["sr"]:
                raise RunRefused(f"{f}: sample rate {sr} != {run['sr']}")
            if len(x) != run["n_samples"]:
                raise RunRefused(f"LENGTH_MISMATCH: {f} has {len(x)} samples, run.json says {run['n_samples']}")
            chans[key] = x
        try:
            events = bi_log.read_log(rd / fev)
            chains = bi_log.validate_log(events)
        except (bi_log.LogRefused, json.JSONDecodeError) as e:
            raise RunRefused(f"EVENT_LOG ({fev}): {e}") from e
        conds[cond] = {"events": events, "chains": chains["cancels"], "ch": chans, "files": (fin, fout, fev)}
    return {"run": run, "cond": conds, "events": conds["overlap"]["events"], "chains": conds["overlap"]["chains"]}


def check_condition(c: dict, sr: int) -> dict:
    ev, ch = c["events"], c["ch"]
    fin, fout, _ = c["files"]
    played = [e for e in ev if e["ev"] == "playback_start"]
    if played and not np.any(ch["output"]):
        raise RunRefused(f"CHANNEL_MISSING: {fout} is all-zero but the log shows playback")
    if np.any(ch["input"]) and np.array_equal(ch["output"], ch["input"]):
        raise RunRefused(f"CHANNEL_DUPLICATED: {fout} is sample-identical to {fin}")
    st = [e for e in ev if e["ev"] == "stim_start"]
    if len(st) != 1:
        raise RunRefused(f"EVENT_LOG: expected exactly one stim_start, found {len(st)}")
    st = st[0]
    base, n = st["stim_base_sample"], st["n_samples"]
    seg = ch["input"][base : base + n]
    if len(seg) != n or A.pcm_sha256(seg) != st["stim_pcm_sha256"]:
        raise RunRefused(f"CLOCK_SKEW: {fin} does not carry the stimulus at the logged sample position")
    if np.any(ch["input"][:base]):
        raise RunRefused(f"CLOCK_SKEW: {fin} has signal before the logged stimulus start")
    if played:
        first = min(e["play_sample"] for e in played)
        nz = np.flatnonzero(ch["output"])
        if len(nz) == 0 or abs(int(nz[0]) - first) > CLOCK_TOL_SAMPLES:
            raise RunRefused(f"CLOCK_SKEW: first non-zero {fout} sample {int(nz[0]) if len(nz) else None} vs logged "
                             f"playback_start sample {first} (tol {CLOCK_TOL_SAMPLES})")
    return {"ok": True, "events": len(ev), "cancel_chains": len(c["chains"])}


def check_integrity(r: dict) -> dict:
    run = r["run"]
    out = {"ok": True, "n_samples": run["n_samples"], "sr": run["sr"]}
    for cond, c in r["cond"].items():
        out[cond] = check_condition(c, run["sr"])
    return out


# ------------------------------------------------------------------ event-log scorer
def eventlog_score(c: dict) -> dict:
    ev = c["events"]
    st = next(e for e in ev if e["ev"] == "stim_start")
    us, ue = st["user_speech_start_ns"], st["user_speech_end_ns"]
    ms = 1e6
    stops = [e for e in ev if e["ev"] == "playback_stop" and e["truncated"] and e["t_ns"] >= us]
    first_stop = stops[0] if stops else None
    starts = [e for e in ev if e["ev"] == "playback_start"]
    turn_kind = {e["turn_id"]: e["kind"] for e in ev if e["ev"] == "turn_start"}
    after_stop = [e for e in starts if first_stop and e["play_ns"] > first_stop["audio_until_ns"]]
    after_user_end = [e for e in starts if e["play_ns"] >= ue]
    chain = None
    if c["chains"]:
        cc = c["chains"][0]
        h = cc["hooks"]
        chain = {
            "late_cancel": cc["late"],
            "cancel_call_ns": cc["call"]["t_ns"],
            "hook_ns": {k: v["t_ns"] for k, v in h.items()},
            "hook_end_ns": {k: v.get("t_end_ns") for k, v in h.items()},
            "order_ok": tuple(h) == bi_log.HOOK_ORDER[: len(h)],
            "chain_latency_ms": None if "orchestrator" not in h else round((h["orchestrator"]["t_end_ns"] - cc["call"]["t_ns"]) / ms, 3),
            "raised": cc["ret"].get("raised"),
        }
    resumed = [e for e in after_stop if turn_kind.get(e["turn_id"]) == "resume"]
    responded = [e for e in after_stop if turn_kind.get(e["turn_id"]) == "respond"]
    if first_stop is None:
        label = "continue"
    elif resumed:
        label = "resume"
    elif responded:
        label = "respond"
    else:
        label = "silent"
    nxt = next((e for e in after_user_end if first_stop is None or e["play_ns"] > first_stop["audio_until_ns"]), None)
    return {
        "label_basis": "event-log deterministic (cancel chain fired / playback resumed / new response generated); "
                       "NON-COMPARABLE to FDB GPT-4o Respond/Resume labels",
        "label": label,
        "chain_fired": bool(chain and chain["order_ok"] and not chain["late_cancel"] and len(chain["hook_ns"]) == 4),
        "playback_resumed": bool(resumed),
        "new_response_generated": bool(responded),
        "user_speech_start_ns": us, "user_speech_end_ns": ue,
        "model_stop_ns": None if first_stop is None else first_stop["audio_until_ns"],
        "t_stop_ms": None if first_stop is None else round((first_stop["audio_until_ns"] - us) / ms, 3),
        "model_restart_ns": None if nxt is None else nxt["play_ns"],
        "t_resp_ms": None if nxt is None else round((nxt["play_ns"] - ue) / ms, 3),
        "cancel_chain": chain,
    }


# ------------------------------------------------------------------ HumDial timing rules
def humdial_score(c: dict, el: dict, vad_model_segments: list[tuple[float, float]]) -> dict:
    """first-response delay = first model audio at/after user segment end - user segment end (event log);
    timing rejection = model audio STARTED (a new turn's first audio, not the already-speaking one) inside the user
    segment, i.e. before the segment end. Both cross-checked against the first Silero model segment starting at/after
    the same reference on output.wav."""
    ev = c["events"]
    us, ue = el["user_speech_start_ns"], el["user_speech_end_ns"]
    first_turn = next((e["turn_id"] for e in ev if e["ev"] == "turn_start"), None)
    starts = [e for e in ev if e["ev"] == "playback_start"]
    after = [e for e in starts if e["play_ns"] >= ue]
    inside = [e for e in starts if us <= e["play_ns"] < ue and e["turn_id"] != first_turn]
    fa = after[0]["play_ns"] if after else None
    cross = next((s for s, _ in vad_model_segments if s * 1e9 >= ue), None)
    out = {
        "definition": "first_response_delay = first model audio start at/after user segment end - user segment end; "
                      "timing_rejection = a new response's audio started before the user segment end",
        "first_model_audio_after_user_end_ns": fa,
        "first_response_delay_ms": None if fa is None else round((fa - ue) / 1e6, 3),
        "timing_rejection": bool(inside),
        "timing_rejection_audio_ns": inside[0]["play_ns"] if inside else None,
        "silero_first_model_segment_start_after_user_end_s": cross,
    }
    if fa is None or cross is None:
        out["crosscheck"] = {"ok": fa is None and cross is None, "reason": "no model audio after user end on both" if fa is None and cross is None else "one side saw model audio, the other did not"}
    else:
        d = cross - fa / 1e9
        out["crosscheck"] = {"ok": abs(d) <= TOL_VAD_S, "silero_minus_eventlog_s": round(d, 3), "tolerance_s": TOL_VAD_S}
    return out


# ------------------------------------------------------------------ FDB v1 adapter
def fdb_v1(r: dict, el: dict, run_dir: Path, asr, scratch: Path, output_name: str) -> dict:
    """Stage the FDB v1 folder layout (ONLY input.wav/output.wav/output.json/turn_taking.json: the unmodified scripts
    glob ``*output.json`` so a ``clean_output.json`` would double-count), word timestamps from the injected ASR."""
    res = asr.transcribe([str(run_dir / output_name)])[0]
    words = res.timestamp["word"]
    chunks = [{"text": w["word"], "timestamp": [float(w["start"]), float(w["end"])]} for w in words]
    root = scratch / "fdb_v1_root"
    if root.exists():
        shutil.rmtree(root)
    fld = root / "item0"
    fld.mkdir(parents=True)
    shutil.copy(run_dir / r["cond"]["overlap"]["files"][0], fld / "input.wav")
    shutil.copy(run_dir / output_name, fld / "output.wav")
    (fld / "output.json").write_text(json.dumps({"chunks": chunks}))
    ue_s = el["user_speech_end_ns"] / 1e9
    (fld / "turn_taking.json").write_text(json.dumps([{"timestamp": [ue_s, ue_s]}]))
    official = {
        "pause_handling": bi_fdb.run_fdb_eval_script("eval_pause_handling.py", root),
        "smooth_turn_taking": bi_fdb.run_fdb_eval_script("eval_smooth_turn_taking.py", root),
    }
    inproc = tor_from_chunks(chunks, ue_s)
    return {"asr_text": res.text, "n_words": len(chunks), "word_timestamps": chunks, "official": official, "inprocess": inproc,
            "measurement_era": "ME-CS48-2"}


def tor_from_chunks(chunks: list[dict], input_end_s: float) -> dict:
    """The exact rule of FDB eval_pause_handling / eval_smooth_turn_taking (duration < 1 s and <= 3 words => no turn)."""
    if not chunks:
        return {"tor": 0, "latency_s": None}
    last = chunks[-1]["timestamp"][-1]
    dur = (last if last is not None else chunks[-1]["timestamp"][0]) - chunks[0]["timestamp"][0]
    tor = 0 if (dur < 1 and len(chunks) <= 3) else 1
    lat = None
    if tor:
        lat = max(0.0, chunks[0]["timestamp"][0] - input_end_s)
    return {"tor": tor, "latency_s": None if lat is None else round(lat, 6)}


# ------------------------------------------------------------------ top level
def score_run(run_dir: str | Path, *, asr=None, cache: str | Path | None = None, silero_model: str | None = None) -> dict:
    rd = Path(run_dir)
    r = load_run(rd)
    integ = check_integrity(r)
    own_tmp = None
    if cache is None:
        own_tmp = tempfile.TemporaryDirectory(prefix="bi-score-")
        cache = own_tmp.name
    cache = Path(cache)
    try:
        gt = bi_fdb.load_get_timing(cache, silero_model)
        fin, fout, _ = r["cond"]["overlap"]["files"]
        cin, cout, _ = r["cond"]["clean"]["files"]
        readings = {}
        for name, g in bi_fdb.GAP_READINGS.items():
            readings[name] = {**g, **bi_fdb.timing_for(gt, rd / fin, rd / fout, g["user_merge_gap_s"], g["model_merge_gap_s"])}
        stage = cache / "fdb_v15_item"
        if stage.exists():
            shutil.rmtree(stage)
        stage.mkdir(parents=True)
        shutil.copy(rd / fin, stage / "input.wav")
        shutil.copy(rd / fout, stage / "output.wav")
        official = bi_fdb.run_official_process_folder(gt, stage)
        readings["code"]["official_process_folder_matches"] = (
            official["latency_stop_list"] == readings["code"]["latency_stop_list"]
            and official["latency_resp_list"] == readings["code"]["latency_resp_list"])
        clean_timing = bi_fdb.timing_for(gt, rd / cin, rd / cout, 0.6, 0.5)
        el = eventlog_score(r["cond"]["overlap"])
        clean_el = eventlog_score(r["cond"]["clean"])
        hd = humdial_score(r["cond"]["overlap"], el, readings["code"]["model_segments"])
        out = {
            "schema": "bi-score-v2",
            "run_sha256": r["run"]["files"],
            "pins": {"silero_sha256": bi_vad.get_vad(silero_model).sha256, "fdb_commit": bi_fdb.COMMIT,
                     "orchestrator_voice_sha256": r["run"]["orchestrator_voice_sha256"]},
            "scenario": {k: r["run"]["scenario"][k] for k in ("id", "overlap_class") if k in r["run"]["scenario"]},
            "integrity": integ,
            "fdb_timing": {"readings": readings,
                           "clean_baseline_code_reading": {k: clean_timing[k] for k in ("user_segments", "model_segments", "t_stop_s", "t_resp_s")},
                           "note": "Silero VAD only; no GPT-4o behavior label, no gpt-4-turbo rating"},
            "eventlog": el,
            "clean_eventlog": {k: clean_el[k] for k in ("label", "chain_fired", "t_stop_ms", "model_stop_ns")},
            "humdial": hd,
            "measurement_era": MEASUREMENT_ERA,
        }
        if asr is not None:
            out["fdb_v1"] = fdb_v1(r, el, rd, asr, cache, fout)
        return out
    finally:
        if own_tmp:
            own_tmp.cleanup()
