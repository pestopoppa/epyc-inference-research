#!/usr/bin/env python3
"""CS-48 overlap session: stimulus player + time-synchronous recorder + cancel-hook timestamps.

The accepted ``VoiceController`` is driven unmodified. The harness injects (a) a ``TimedAudioQueue`` as its
``AudioQueue`` (the "queue" cancel hook, playback simulation, and the recorder), and (b) a ``TimedBackend`` wrapper
around the cascade so that ``cancel_generation`` / ``cancel_vocoder`` / ``cancel_orchestrator`` carry timestamps.
Order and exactly-once semantics are the controller's own; the log only observes them.

Two conditions per scenario, exactly as Full-Duplex-Bench v1.5 lays them out: the OVERLAP condition streams
``input.wav`` (user audio WITH the overlap event) and records ``output.wav``; the CLEAN condition streams
``clean_input.wav`` (the same user audio WITHOUT the overlap event) and records ``clean_output.wav`` -- the model's
response to the non-overlap reference, which FDB uses as the baseline for adaptation analyses. Each condition has its own
event log (``events.jsonl`` / ``clean_events.jsonl``); all six files share one sample count and one t0 anchor.

Time base: every event is ``t_ns`` since run t0; every wav is anchored so sample 0 == t0 and ``sample = round(t_ns *
sr / 1e9)``. All three wavs share one length. Playback is accounted in integer samples, so a virtual-clock run is
exact (conformance) and a real-clock run is exact up to the scheduler's own wake-up jitter, which is in the log.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path

import numpy as np

import bi_audio as A
import bi_orch
from bi_log import EventLog

FRAME_S = 0.020


def ns_to_sample(t_ns: int, sr: int) -> int:
    return (t_ns * sr * 2 + 1_000_000_000) // 2_000_000_000  # round-half-up, integer-exact


def sample_to_ns(n: int, sr: int) -> int:
    return n * 1_000_000_000 // sr


class TimedAudioQueue:
    """AudioQueue + playback simulator + recorder for the model channel."""

    def __init__(self, clock, log: EventLog, sr: int, model_sr: int) -> None:
        self.clock, self.log, self.sr, self.model_sr = clock, log, sr, model_sr
        self.segments: list[list] = []  # [start_sample, end_sample, int16 array, turn_id]
        self.tail = 0
        self.lock = threading.Lock()
        self._started: set[str] = set()

    def enqueue(self, turn_id: str, pcm: bytes) -> None:
        x = np.frombuffer(pcm, dtype="<i2")
        y = A.resample_int16(x, self.model_sr, self.sr)
        now_s = ns_to_sample(self.clock.now_ns() - self.log.t0, self.sr)
        with self.lock:
            start = max(now_s, self.tail)
            self.segments.append([start, start + len(y), y, turn_id])
            self.tail = start + len(y)
            first = turn_id not in self._started
            self._started.add(turn_id)
        if first:
            self.log.emit("playback_start", turn_id=turn_id, play_ns=sample_to_ns(start, self.sr), play_sample=start)
        self.log.emit("audio_enqueue", turn_id=turn_id, start_sample=start, n_samples=len(y))

    def stop_queued_audio(self, session_id: str, turn_id: str) -> None:
        rec = self.log.emit("hook", hook="queue", session_id=session_id, turn_id=turn_id)
        cut = ns_to_sample(rec["t_ns"], self.sr)
        with self.lock:
            active = any(seg[1] > cut for seg in self.segments)
            until = cut
            for seg in self.segments:
                if seg[1] > cut:
                    if seg[0] < cut:
                        seg[2] = seg[2][: cut - seg[0]]
                        seg[1] = cut
                    else:
                        seg[1] = seg[0]
                        seg[2] = seg[2][:0]
            self.tail = min(self.tail, cut)
            played = [s for s in self.segments if s[1] > s[0]]
            if played:
                until = min(cut, max(s[1] for s in played))
        rec["t_end_ns"] = self.clock.now_ns() - self.log.t0
        self.log.emit("playback_stop", turn_id=turn_id, truncated=bool(active), audio_until_ns=sample_to_ns(until, self.sr),
                      audio_until_sample=until)

    def render(self, n: int) -> np.ndarray:
        out = np.zeros(n, dtype=np.int32)
        for s, e, y, _ in self.segments:
            if e > s:
                out[s:e] += y[: e - s].astype(np.int32)
        return np.clip(out, -32768, 32767).astype("<i2")


class TimedBackend:
    """Wrap a VoiceBackend so each cancel hook logs a timestamp. Pure observation; behavior is unchanged."""

    def __init__(self, inner, log: EventLog) -> None:
        self._inner, self._log = inner, log

    def _hook(self, name: str, fn) -> None:
        rec = self._log.emit("hook", hook=name)
        try:
            fn()
        finally:
            rec["t_end_ns"] = self._log.clock.now_ns() - self._log.t0

    def cancel_generation(self) -> None:
        self._hook("generation", self._inner.cancel_generation)

    def cancel_vocoder(self) -> None:
        self._hook("vocoder", self._inner.cancel_vocoder)

    def cancel_orchestrator(self) -> None:
        self._hook("orchestrator", self._inner.cancel_orchestrator)

    def __getattr__(self, item):  # ingest / respond / inject / health pass through
        return getattr(self._inner, item)


class NeverCancel:
    def run(self, sess) -> None:
        return None

    def on_frame(self, sess, t_ns: int, frame: np.ndarray) -> None:
        return None


class ScriptedPolicy:
    """Exact-time actions for conformance. steps: [{"at_ns": int (since t0), "do": "cancel"|"turn", ...}]."""

    def __init__(self, steps: list[dict]) -> None:
        self.steps = steps

    def run(self, sess) -> None:
        for st in self.steps:
            sess.clock.sleep_until_ns(sess.log.t0 + st["at_ns"])
            if st["do"] == "cancel":
                sess.cancel()
            elif st["do"] == "turn":
                sess.run_turn(st["turn_id"], kind=st["kind"], transcript=st.get("transcript", "continue"))

    def on_frame(self, sess, t_ns, frame) -> None:
        return None


class Session:
    """One condition (overlap or clean) of one scenario."""

    def __init__(self, clock, scenario: dict, backend_factory, policy, *, condition: str = "overlap", sr: int = A.SR,
                 model_sr: int = A.SR, session_id: str = "bi-session", first_turn_id: str = "t0", agent_offset_s: float = 0.0) -> None:
        VoiceController, CascadeBackend, self.VoiceTurn, self.VoiceEvent = bi_orch.load_voice()
        assert condition in ("overlap", "clean")
        self.clock, self.scenario, self.policy, self.sr, self.condition = clock, scenario, policy, sr, condition
        self.session_id, self.first_turn_id, self.agent_offset_s = f"{session_id}-{condition}", first_turn_id, agent_offset_s
        self.t0 = clock.now_ns()
        self.log = EventLog(clock, self.t0)
        self.queue = TimedAudioQueue(clock, self.log, sr, model_sr)
        self.backend = TimedBackend(backend_factory(clock, self.log), self.log)
        self.controller = VoiceController(self.backend, self.queue)
        wav = scenario["stimulus_wav"] if condition == "overlap" else scenario.get("clean_stimulus_wav")
        ref, ref_sr = A.read_wav(scenario["stimulus_wav"])
        if wav is None:  # no clean reference supplied: digital silence of the same length (no user at all)
            self.stim, self.stim_sr = np.zeros_like(ref), ref_sr
        else:
            self.stim, self.stim_sr = A.read_wav(wav)
        if self.stim_sr != sr:
            self.stim = A.resample_int16(self.stim, self.stim_sr, sr)
        self.input_buf = np.zeros(0, dtype="<i2")
        self._live_turns: set[str] = set()
        self._cur_turn: str | None = None

    # ---- harness-side actions -------------------------------------------------
    def run_turn(self, turn_id: str, *, kind: str, transcript: str = "respond") -> None:
        turn = self.VoiceTurn(turn_id=turn_id, session_id=self.session_id, audio_chunk=b"", transcript=transcript,
                              metadata={"harness_kind": kind})
        self.log.emit("turn_start", turn_id=turn_id, kind=kind)
        self._live_turns.add(turn_id)
        self._cur_turn = turn_id
        terminal = None
        for ev in self.controller.respond(turn):
            if ev.kind == "audio_chunk":
                self.queue.enqueue(turn_id, ev.payload)
            elif ev.kind in ("end", "error"):
                terminal = ev.kind
                self.log.emit("turn_" + ev.kind, turn_id=turn_id, detail=ev.payload if ev.kind == "error" else None)
            else:
                self.log.emit("backend_event", turn_id=turn_id, kind=ev.kind)
        self._live_turns.discard(turn_id)
        if terminal is None:
            self.log.emit("turn_cancelled", turn_id=turn_id)

    def cancel(self, turn_id: str | None = None) -> None:
        tid = turn_id or self._cur_turn
        live = tid in self._live_turns
        self.log.emit("cancel_call", turn_id=tid, turn_live=live)
        raised = None
        try:
            self.controller.cancel(self.session_id, tid)
        except RuntimeError as e:
            raised = str(e)
        self.log.emit("cancel_return", turn_id=tid, raised=raised)

    # ---- stimulus player ---------------------------------------------------------
    def _player(self) -> None:
        sc, sr = self.scenario, self.sr
        offset_ns = int(round(sc["stimulus_offset_s"] * 1e9))
        n = len(self.stim)
        step = int(round(FRAME_S * sr))
        us0 = offset_ns + int(round(sc["user_speech_start_s"] * 1e9))
        us1 = offset_ns + int(round(sc["user_speech_end_s"] * 1e9))
        self.clock.sleep_until_ns(self.t0 + offset_ns)
        self.log.emit("stim_start", scenario_id=sc["id"], condition=self.condition, overlap_class=sc["overlap_class"],
                      user_speech_start_ns=us0, user_speech_end_ns=us1, stimulus_offset_ns=offset_ns, n_samples=n,
                      stim_pcm_sha256=A.pcm_sha256(self.stim), stim_base_sample=ns_to_sample(offset_ns, sr))
        base = ns_to_sample(offset_ns, sr)
        self.input_buf = np.zeros(base + n, dtype="<i2")
        for k in range(0, n, step):
            self.clock.sleep_until_ns(self.t0 + sample_to_ns(base + k, sr))
            fr = self.stim[k : k + step]
            self.input_buf[base + k : base + k + len(fr)] = fr
            self.policy.on_frame(self, self.clock.now_ns() - self.t0, fr)
        self.log.emit("stim_end", scenario_id=sc["id"], condition=self.condition)

    def execute(self) -> None:
        self.t0 = self.log.t0 = self.clock.now_ns()  # t0 == the instant tasks are released, not object construction
        self.clock.spawn("agent", lambda: (self.clock.sleep_until_ns(self.t0 + int(round(self.agent_offset_s * 1e9))),
                                           self.run_turn(self.first_turn_id, kind="initial")))
        self.clock.spawn("player", self._player)
        self.clock.spawn("policy", lambda: self.policy.run(self))
        self.clock.run_all()

    def min_samples(self, tail_s: float) -> int:
        end_ns = max(e["t_ns"] for e in self.log.events)
        return max(len(self.input_buf), self.queue.tail, ns_to_sample(end_ns, self.sr)) + int(tail_s * self.sr)

    def render(self, n: int) -> tuple[np.ndarray, np.ndarray]:
        inp = np.zeros(n, dtype="<i2")
        inp[: len(self.input_buf)] = self.input_buf
        return inp, self.queue.render(n)


def run_pair(scenario: dict, backend_factory, policy_factory, out_dir: str | Path, *, clock_factory, sr: int = A.SR,
             model_sr: int = A.SR, agent_offset_s: float = 0.0, tail_s: float = 1.0, extra_cfg: dict | None = None) -> dict:
    """Record both FDB v1.5 conditions into ``out_dir`` (input/output/events and clean_input/clean_output/clean_events).

    ``policy_factory(condition)`` returns a fresh policy per condition. Conditions run one after the other (each in its
    own clock domain, each real-time on a RealClock), then are padded to one common sample count.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sess = {}
    for cond in ("overlap", "clean"):
        sess[cond] = Session(clock_factory(), scenario, backend_factory, policy_factory(cond), condition=cond, sr=sr,
                             model_sr=model_sr, agent_offset_s=agent_offset_s)
        sess[cond].execute()
    n = max(s.min_samples(tail_s) for s in sess.values())
    names = {"overlap": ("input.wav", "output.wav", "events.jsonl"), "clean": ("clean_input.wav", "clean_output.wav", "clean_events.jsonl")}
    for cond, s in sess.items():
        a, b, c = names[cond]
        inp, outp = s.render(n)
        A.write_wav(out / a, inp, sr)
        A.write_wav(out / b, outp, sr)
        s.log.write(out / c)
    run = {
        "schema": "bi-run-v2", "clock": sess["overlap"].clock.kind, "sr": sr, "n_samples": int(n), "scenario": scenario,
        "conditions": names, "agent_offset_s": agent_offset_s,
        "files": {f: A.file_sha256(out / f) for names_ in names.values() for f in names_},
        "orchestrator_voice_sha256": bi_orch.voice_hashes(), "orchestrator_root": str(bi_orch.orch_root()), "cfg": extra_cfg or {},
    }
    (out / "run.json").write_text(json.dumps(run, sort_keys=True, indent=1) + "\n")
    return run
