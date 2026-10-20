"""CS-48 conformance suite: known-answer (scripted fake agent), mutation, and bit-identical-replay checks.

Run: <orchestrator venv python> -I -m pytest scripts/voice/barge_in/tests -q   (CPU-light, no service, no network)
Declared tolerances: event-log quantities are EXACT on the virtual clock (0 ns); Silero-derived quantities are
recovered within TOL_SILERO_S (a 32 ms window + 30 ms pad + onset lag bound; the observed maximum is printed).
"""
import json
import shutil
import numpy as np
import pytest

import bi_audio as A
import bi_clock
import bi_fake
import bi_fixtures as F
import bi_log
import bi_score
import bi_session as S
import bi_vad

TOL_SILERO_S = 0.15
NS = 1_000_000_000


@pytest.fixture(scope="session")
def scen(tmp_path_factory):
    return {s["id"]: s for s in F.materialise(tmp_path_factory.mktemp("stim"))}


@pytest.fixture(scope="session")
def cache(tmp_path_factory):
    return tmp_path_factory.mktemp("cache")


def abs_ns(sc, which, plus_s=0.0):
    return int(round((sc["stimulus_offset_s"] + sc[which]) * NS)) + int(round(plus_s * NS))


def record(out, sc, steps, *, model_sr=16000, clock_factory=bi_clock.VirtualClock, clean_steps=(), pcm=None, **kw):
    pcm = F.agent_clip(model_sr) if pcm is None else pcm
    holder = []
    return S.run_pair(sc, bi_fake.make_factory(pcm, model_sr, holder),
                      lambda cond: S.ScriptedPolicy(list(steps) if cond == "overlap" else list(clean_steps)), out,
                      clock_factory=clock_factory, model_sr=model_sr, **kw), holder


def interrupt_steps(sc, stop_after_s=0.4, respond_after_s=0.5):
    return [{"at_ns": abs_ns(sc, "user_speech_start_s", stop_after_s), "do": "cancel"},
            {"at_ns": abs_ns(sc, "user_speech_end_s", respond_after_s), "do": "turn", "turn_id": "t1", "kind": "respond"}]


# ------------------------------------------------------------------ (d) known answer
def test_known_answer_interruption(tmp_path, scen, cache):
    sc = scen["interruption-01"]
    run, fakes = record(tmp_path / "r", sc, interrupt_steps(sc))
    assert fakes[0].cancel_log == ["generation", "vocoder", "orchestrator"]  # the fake's OWN record of the controller's hook order
    assert fakes[1].cancel_log == []  # clean condition: nothing cancelled
    s = bi_score.score_run(tmp_path / "r", cache=cache)
    el = s["eventlog"]
    assert el["t_stop_ms"] == 400.0 and el["t_resp_ms"] == 500.0  # exact
    assert el["label"] == "respond" and el["chain_fired"] and el["new_response_generated"] and not el["playback_resumed"]
    ch = el["cancel_chain"]
    assert ch["order_ok"] and list(ch["hook_ns"]) == ["queue", "generation", "vocoder", "orchestrator"]
    assert all(v == abs_ns(sc, "user_speech_start_s", 0.4) for v in ch["hook_ns"].values())  # all four at the scripted instant
    for name, rd in s["fdb_timing"]["readings"].items():
        assert abs(rd["t_stop_s"][0] - 0.400) <= TOL_SILERO_S, name
        assert abs(rd["t_resp_s"][0] - 0.500) <= 2 * TOL_SILERO_S, name
    assert s["fdb_timing"]["readings"]["code"]["official_process_folder_matches"]
    hd = s["humdial"]
    assert hd["first_response_delay_ms"] == 500.0 and hd["timing_rejection"] is False and hd["crosscheck"]["ok"]
    print("observed silero t_stop err", s["fdb_timing"]["readings"]["code"]["t_stop_s"][0] - 0.4,
          "t_resp err", s["fdb_timing"]["readings"]["code"]["t_resp_s"][0] - 0.5, "hd", hd["crosscheck"])


@pytest.mark.parametrize("stop_after", [0.10, 0.25, 0.70, 1.20])
def test_stop_grid(tmp_path, scen, cache, stop_after):
    sc = scen["interruption-01"]
    record(tmp_path / "r", sc, interrupt_steps(sc, stop_after))
    s = bi_score.score_run(tmp_path / "r", cache=cache)
    assert s["eventlog"]["t_stop_ms"] == round(stop_after * 1000, 3)
    assert abs(s["fdb_timing"]["readings"]["code"]["t_stop_s"][0] - stop_after) <= TOL_SILERO_S


def test_resume_label_and_timing_rejection(tmp_path, scen, cache):
    sc = scen["interruption-01"]
    steps = [{"at_ns": abs_ns(sc, "user_speech_start_s", 0.3), "do": "cancel"},
             {"at_ns": abs_ns(sc, "user_speech_end_s", 0.8), "do": "turn", "turn_id": "t1", "kind": "resume"}]
    record(tmp_path / "a", sc, steps)
    el = bi_score.score_run(tmp_path / "a", cache=cache)["eventlog"]
    assert el["label"] == "resume" and el["playback_resumed"] and not el["new_response_generated"] and el["t_resp_ms"] == 800.0
    early = [{"at_ns": abs_ns(sc, "user_speech_start_s", 0.3), "do": "cancel"},
             {"at_ns": abs_ns(sc, "user_speech_start_s", 1.5), "do": "turn", "turn_id": "t1", "kind": "respond"}]
    record(tmp_path / "b", sc, early)
    hd = bi_score.score_run(tmp_path / "b", cache=cache)["humdial"]
    assert hd["timing_rejection"] is True and hd["timing_rejection_audio_ns"] == abs_ns(sc, "user_speech_start_s", 1.5)


@pytest.mark.parametrize("cid", ["backchannel-01", "talking-to-others-01", "background-speech-01"])
def test_no_stop_classes(tmp_path, scen, cache, cid):
    sc = scen[cid]
    record(tmp_path / "r", sc, [])
    s = bi_score.score_run(tmp_path / "r", cache=cache)
    el = s["eventlog"]
    assert el["label"] == "continue" and el["t_stop_ms"] is None and el["cancel_chain"] is None and not el["chain_fired"]
    # FDB reports the whole overlap as 't_stop' when the model never stops (a HIGH t_stop is CORRECT for these classes)
    rd = s["fdb_timing"]["readings"]["code"]
    ends = [e for _, e in rd["user_segments"]]
    assert rd["latency_stop_list"] and all(e in ends for _, e in rd["latency_stop_list"])  # overlap ends where the USER stops


def test_late_cancel_is_queue_only(tmp_path, scen, cache):
    sc = scen["backchannel-01"]
    pcm = F.agent_clip(16000)[: 16000 * 2]  # agent finishes at ~1 s, long before the user cuts in
    record(tmp_path / "r", sc, [{"at_ns": abs_ns(sc, "user_speech_start_s", 0.2), "do": "cancel"}], pcm=pcm)
    s = bi_score.score_run(tmp_path / "r", cache=cache)
    assert s["eventlog"]["cancel_chain"]["late_cancel"] and not s["eventlog"]["chain_fired"] and s["eventlog"]["t_stop_ms"] is None


def test_both_gap_readings_differ_where_they_should(tmp_path, cache):
    """Pause of 0.25 s of digital zero (+ the clips' own edge silence): merged under the CODE user gap 0.6 s, split under the COMMENT gap 0.3 s."""
    seg1, seg2 = F._load("121-121726-0006")[int(0.40 * 16000):int(1.40 * 16000)], F._load("121-121726-0011")[int(0.5 * 16000):int(1.3 * 16000)]
    x = np.concatenate([seg1, np.zeros(int(0.25 * 16000), dtype="<i2"), seg2, np.zeros(4800, dtype="<i2")])
    p = tmp_path / "pause.wav"
    A.write_wav(p, x)
    s0, e0 = F.annotate(x)
    sc = {"id": "pause-01", "overlap_class": "user_interruption", "stimulus_wav": str(p), "stimulus_offset_s": 3.0,
          "user_speech_start_s": s0, "user_speech_end_s": e0}
    record(tmp_path / "r", sc, [])
    rd = bi_score.score_run(tmp_path / "r", cache=cache)["fdb_timing"]["readings"]
    assert len(rd["code"]["user_segments"]) == 1 and len(rd["comment"]["user_segments"]) == 2
    assert (rd["code"]["user_merge_gap_s"], rd["code"]["model_merge_gap_s"]) == (0.6, 0.5)
    assert (rd["comment"]["user_merge_gap_s"], rd["comment"]["model_merge_gap_s"]) == (0.3, 0.05)


def test_24k_model_rate_and_clean_baseline(tmp_path, scen, cache):
    sc = scen["interruption-01"]
    record(tmp_path / "a", sc, interrupt_steps(sc), model_sr=24000)
    s = bi_score.score_run(tmp_path / "a", cache=cache)
    assert s["eventlog"]["t_stop_ms"] == 400.0
    out, _ = A.read_wav(tmp_path / "a" / "output.wav")
    cl, _ = A.read_wav(tmp_path / "a" / "clean_output.wav")
    assert not np.array_equal(out, cl)  # overlap run was cut at 3.53 s; the clean-reference run keeps talking
    assert s["clean_eventlog"]["label"] == "continue" and s["clean_eventlog"]["t_stop_ms"] is None
    cb = s["fdb_timing"]["clean_baseline_code_reading"]
    assert cb["model_segments"][0][1] > 6.0  # clean_output.wav: the model talked on through the (absent) interruption


# ------------------------------------------------------------------ (d) real clock smoke
def test_real_clock_realtime_player(tmp_path, scen, cache):
    import time
    sc = dict(scen["backchannel-01"], stimulus_offset_s=1.0)
    steps = [{"at_ns": abs_ns(sc, "user_speech_start_s", 0.3), "do": "cancel"}]
    t = time.monotonic()
    record(tmp_path / "r", sc, steps, clock_factory=bi_clock.RealClock)
    wall = time.monotonic() - t
    assert 4.0 <= wall < 25.0  # both conditions really ran in real time (the un-cancelled clean agent streams ~8.8 s at 2x pace)
    s = bi_score.score_run(tmp_path / "r", cache=cache)
    el = s["eventlog"]
    assert el["chain_fired"] and abs(el["t_stop_ms"] - 300.0) < 30.0, el["t_stop_ms"]  # wake-up jitter bound
    print("real-clock t_stop_ms", el["t_stop_ms"], "chain_ms", el["cancel_chain"]["chain_latency_ms"])


# ------------------------------------------------------------------ (d) mutation cases
def _rehash(rd):
    run = json.loads((rd / "run.json").read_text())
    for f in run["files"]:
        if (rd / f).exists():
            run["files"][f] = A.file_sha256(rd / f)
    (rd / "run.json").write_text(json.dumps(run, sort_keys=True, indent=1) + "\n")


@pytest.fixture()
def good(tmp_path, scen):
    sc = scen["interruption-01"]
    record(tmp_path / "good", sc, interrupt_steps(sc))
    return tmp_path / "good"


def mutated(good, tmp_path, name):
    d = tmp_path / name
    shutil.copytree(good, d)
    return d


def refuse(d, cache, match):
    with pytest.raises(bi_score.RunRefused, match=match):
        bi_score.score_run(d, cache=cache)


def test_baseline_good_is_accepted(good, cache):
    assert bi_score.score_run(good, cache=cache)["integrity"]["ok"]


def test_mutation_altered_clip(good, tmp_path, cache):
    d = mutated(good, tmp_path, "m")
    x, sr = A.read_wav(d / "output.wav")
    x[5000] ^= 1
    A.write_wav(d / "output.wav", x, sr)
    refuse(d, cache, "HASH_MISMATCH")


def test_mutation_shifted_output_audio(good, tmp_path, cache):
    d = mutated(good, tmp_path, "m")
    for f in ("output.wav", "clean_output.wav"):
        x, sr = A.read_wav(d / f)
        A.write_wav(d / f, np.roll(x, 4000), sr)  # 250 ms late relative to the logged clock
    _rehash(d)
    refuse(d, cache, "CLOCK_SKEW")


def test_mutation_shifted_event_clock(good, tmp_path, cache):
    d = mutated(good, tmp_path, "m")
    ev = bi_log.read_log(d / "events.jsonl")
    for e in ev:
        e["t_ns"] += 250_000_000
        for k in ("play_ns", "audio_until_ns"):
            if k in e:
                e[k] += 250_000_000
        for k in ("play_sample", "audio_until_sample"):
            if k in e:
                e[k] += 4000
        for k in ("user_speech_start_ns", "user_speech_end_ns"):
            if k in e:
                e[k] += 250_000_000
    (d / "events.jsonl").write_text("".join(json.dumps(e, sort_keys=True, separators=(",", ":")) + "\n" for e in ev))
    _rehash(d)
    refuse(d, cache, "CLOCK_SKEW")


def test_mutation_missing_and_silent_output(good, tmp_path, cache):
    d = mutated(good, tmp_path, "gone")
    (d / "output.wav").unlink()
    refuse(d, cache, "CHANNEL_MISSING")
    d2 = mutated(good, tmp_path, "silent")
    x, sr = A.read_wav(d2 / "output.wav")
    A.write_wav(d2 / "output.wav", np.zeros_like(x), sr)
    _rehash(d2)
    refuse(d2, cache, "CHANNEL_MISSING")


def test_mutation_clean_channel(good, tmp_path, cache):
    d = mutated(good, tmp_path, "cm")
    (d / "clean_output.wav").unlink()
    refuse(d, cache, "CHANNEL_MISSING")
    d = mutated(good, tmp_path, "cd")
    x, sr = A.read_wav(d / "clean_input.wav")
    x[1000:2000] = 700  # a non-silent clean input (the fixture's is digital silence)
    A.write_wav(d / "clean_input.wav", x, sr)
    A.write_wav(d / "clean_output.wav", x, sr)
    _rehash(d)
    refuse(d, cache, "CHANNEL_DUPLICATED")
    d = mutated(good, tmp_path, "cs")
    x, sr = A.read_wav(d / "clean_output.wav")
    A.write_wav(d / "clean_output.wav", np.roll(x, 3000), sr)
    _rehash(d)
    refuse(d, cache, "CLOCK_SKEW")


def test_mutation_duplicated_channel(good, tmp_path, cache):
    d = mutated(good, tmp_path, "dup")
    shutil.copy(d / "input.wav", d / "output.wav")
    _rehash(d)
    refuse(d, cache, "CHANNEL_DUPLICATED")


def test_mutation_event_log_defects(good, tmp_path, cache):
    ev = bi_log.read_log(good / "events.jsonl")

    def write(d, evs):
        (d / "events.jsonl").write_text("".join(json.dumps(e, sort_keys=True, separators=(",", ":")) + "\n" for e in evs))
        _rehash(d)

    d = mutated(good, tmp_path, "dupline")
    k = next(i for i, e in enumerate(ev) if e["ev"] == "hook")
    write(d, ev[: k + 1] + [ev[k]] + ev[k + 1 :])
    refuse(d, cache, "EVENT_LOG")
    d = mutated(good, tmp_path, "droppedhook")
    write(d, [dict(e, i=j) for j, e in enumerate([e for e in ev if e.get("hook") != "vocoder"])])
    refuse(d, cache, "EVENT_LOG.*out of order|EVENT_LOG.*hook")
    d = mutated(good, tmp_path, "reordered")
    hs = [i for i, e in enumerate(ev) if e["ev"] == "hook"]
    ev2 = [dict(e) for e in ev]
    ev2[hs[1]], ev2[hs[2]] = dict(ev[hs[2]], i=ev[hs[1]]["i"]), dict(ev[hs[1]], i=ev[hs[2]]["i"])
    write(d, ev2)
    refuse(d, cache, "EVENT_LOG.*out of order")
    d = mutated(good, tmp_path, "backwards")
    ev3 = [dict(e) for e in ev]
    ev3[-1]["t_ns"] = 0
    write(d, ev3)
    refuse(d, cache, "EVENT_LOG.*backwards")


def test_mutation_wrong_silero_hash_refused(tmp_path, good, cache):
    bad = tmp_path / "silero_bad.onnx"
    shutil.copy(bi_vad.tt("tt_silero").DEFAULT_MODEL, bad)
    with open(bad, "ab") as f:
        f.write(b"\0")
    pm = bi_vad.tt("tt_silero").PinMismatch
    with pytest.raises(pm):
        bi_vad.get_vad(str(bad))
    with pytest.raises(pm):  # the FDB scorer path refuses too (patched get_timing builds PinnedSilero at import)
        bi_score.score_run(good, cache=cache, silero_model=str(bad))


def test_mutation_wrong_fdb_base_refused(tmp_path, monkeypatch, good, cache):
    import bi_fdb
    fake = tmp_path / "fdb"
    p = fake / "v1_v1.5/evaluation/get_timing.py"
    p.parent.mkdir(parents=True)
    shutil.copy(bi_fdb.fdb_repo() / "v1_v1.5/evaluation/get_timing.py", p)
    p.write_text(p.read_text() + "\n# tampered\n")
    monkeypatch.setenv("FDB_REPO", str(fake))
    with pytest.raises(bi_fdb.PinRefused):
        bi_score.score_run(good, cache=tmp_path / "c2")


# ------------------------------------------------------------------ (d) bit-identical replay
def test_replay_bit_identical(tmp_path, scen, cache):
    sc = scen["interruption-01"]
    a, _ = record(tmp_path / "a", sc, interrupt_steps(sc))
    b, _ = record(tmp_path / "b", sc, interrupt_steps(sc))
    assert a["files"] == b["files"] and len(a["files"]) == 6  # two recordings: identical wav and event-log bytes
    s1 = bi_score.canonical_bytes(bi_score.score_run(tmp_path / "a", cache=cache))
    s2 = bi_score.canonical_bytes(bi_score.score_run(tmp_path / "a", cache=cache))  # same recorded run scored twice
    s3 = bi_score.canonical_bytes(bi_score.score_run(tmp_path / "b", cache=tmp_path / "other-cache"))
    assert s1 == s2 == s3


# ------------------------------------------------------------------ provenance: controller consumed, not modified
def test_orchestrator_source_unmodified_and_recorded(tmp_path, scen):
    import subprocess, bi_orch
    sc = scen["interruption-01"]
    run, _ = record(tmp_path / "r", sc, [])
    assert set(run["orchestrator_voice_sha256"]) >= {"controller.py", "cascade.py", "contracts.py"}
    root = bi_orch.orch_root()
    r = subprocess.run(["git", "-C", str(root), "status", "--porcelain", "--", "src/voice"], capture_output=True, text=True)
    if r.returncode == 0:
        assert r.stdout.strip() == ""


# ------------------------------------------------------------------ FDB v1 adapter (stub ASR; live whisper.cpp optional)
class StubASR:
    def __init__(self, words):
        self.words = words

    def transcribe(self, paths):
        class R:
            pass
        r = R()
        r.text = " ".join(w[0] for w in self.words)
        r.timestamp = {"word": [{"word": w, "start": s, "end": e} for w, s, e in self.words]}
        return [r]


def test_fdb_v1_adapter_matches_unmodified_scripts(tmp_path, scen, cache):
    sc = scen["interruption-01"]
    record(tmp_path / "r", sc, interrupt_steps(sc))
    ue = sc["stimulus_offset_s"] + sc["user_speech_end_s"]
    words = [("so", ue + 0.5, ue + 0.7), ("here", ue + 0.7, ue + 0.9), ("is", ue + 0.9, ue + 1.1), ("the", ue + 1.1, ue + 1.3), ("plan", ue + 1.3, ue + 1.8)]
    s = bi_score.score_run(tmp_path / "r", cache=cache, asr=StubASR(words))["fdb_v1"]
    assert s["inprocess"] == {"tor": 1, "latency_s": 0.5}
    assert s["official"]["smooth_turn_taking"] == {"average_take_turn": 1.0, "average_latency_s": 0.5}
    assert s["official"]["pause_handling"] == {"average_take_turn": 1.0}
    short = bi_score.score_run(tmp_path / "r", cache=cache, asr=StubASR(words[:2]))["fdb_v1"]  # <1 s and <=3 words => no turn
    assert short["inprocess"]["tor"] == 0 and short["official"]["pause_handling"]["average_take_turn"] == 0.0
    assert "script_error" in short["official"]["smooth_turn_taking"]  # FDB divides by zero when no item takes the turn
    none = bi_score.score_run(tmp_path / "r", cache=cache, asr=StubASR([]))["fdb_v1"]
    assert none["inprocess"] == {"tor": 0, "latency_s": None} and none["official"]["pause_handling"]["average_take_turn"] == 0.0


@pytest.mark.skipif(not __import__("os").environ.get("BI_LIVE_ASR"), reason="live whisper.cpp wiring check: set BI_LIVE_ASR=1, run under region-lock")
def test_live_pinned_whisper_cpp_wiring(tmp_path, scen, cache):
    asr = bi_vad.tt("tt_asr_words").PinnedWhisperCpp(threads=4)
    sc = scen["interruption-01"]
    record(tmp_path / "r", sc, interrupt_steps(sc))
    s = bi_score.score_run(tmp_path / "r", cache=cache, asr=asr)["fdb_v1"]
    assert s["n_words"] > 5
    print("live asr words", s["n_words"], s["inprocess"], s["official"])


# ------------------------------------------------------------------ scenario sources + "no judge, no service"
def test_manifest_adapters(tmp_path, scen):
    import bi_manifest as M
    with pytest.raises(M.ManifestRefused, match="manual Google Drive fetch pending"):
        M.scenarios_from_fdb15(tmp_path / "absent")
    sd = tmp_path / "fdb15" / "talking_to_other" / "s001"
    sd.mkdir(parents=True)
    shutil.copy(scen["talking-to-others-01"]["stimulus_wav"], sd / "input.wav")
    shutil.copy(scen["talking-to-others-01"]["stimulus_wav"], sd / "clean_input.wav")
    (sd / "metadata.json").write_text(json.dumps({"context_text": "x", "current_turn_text": "y", "timestamps": [0.1, 3.3]}))
    got = M.scenarios_from_fdb15(tmp_path / "fdb15")
    assert got[0]["overlap_class"] == "talking_to_others" and got[0]["user_speech_end_s"] == 3.3 and "clean_stimulus_wav" in got[0]
    row = {"id": "c45-1", "overlap_class": "backchannel", "user_wav": scen["backchannel-01"]["stimulus_wav"],
           "user_speech_start_s": 0.1, "user_speech_end_s": 0.9,
           "pcm_sha256": scen["backchannel-01"]["stimulus_pcm_sha256"]}
    mf = tmp_path / "m.jsonl"
    mf.write_text(json.dumps(row) + "\n")
    assert M.scenarios_from_cs45(mf)[0]["id"] == "c45-1"
    mf.write_text(json.dumps(dict(row, pcm_sha256="0" * 64)) + "\n")
    with pytest.raises(M.ManifestRefused, match="PCM-SHA-256"):
        M.scenarios_from_cs45(mf)


def test_no_cloud_judge_no_network_surface(tmp_path, scen, cache, monkeypatch):
    import socket
    import sys
    from pathlib import Path

    def no_connect(self, *a, **k):
        raise AssertionError("the harness opened a network connection")

    monkeypatch.setattr(socket.socket, "connect", no_connect)
    monkeypatch.setattr(socket.socket, "connect_ex", no_connect)
    sc = scen["interruption-01"]
    record(tmp_path / "r", sc, interrupt_steps(sc))
    bi_score.score_run(tmp_path / "r", cache=cache, asr=StubASR([("a", 4.0, 4.1)]))
    for banned in ("openai", "google.generativeai", "eval_behavior", "eval_user_interruption", "torch"):
        assert banned not in sys.modules, banned
    src = "".join(p.read_text() for p in Path(__file__).resolve().parent.parent.glob("bi_*.py"))
    for tok in ("OPENAI", "api.openai", "localhost:9000", "localhost:9002", "localhost:8000", "pkill"):
        assert tok not in src, tok
