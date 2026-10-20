"""CS-46 (c)/(d): chunked greedy-RNNT Parakeet-Realtime-EOU-120M arm on onnxruntime CPU.

One interface for every CS-47 turn-end arm::

    arm.feed(pcm_chunk: float32 mono 16 kHz) -> list[{"t": float, "kind": str, ...}]

Event kinds
  ``token``  non-blank, non-EOU/EOB token (``id``, ``piece``)
  ``eou``    the model emitted ``<EOU>`` - SURFACED explicitly under every reset policy
             (parakeet-rs with reset_on_eou=false only breaks the decode loop, emits nothing)
  ``eob``    ``<EOB>`` (end of backchannel)
  ``reset``  a reset policy fired (``policy`` = decoder|full)

``t`` is STREAM time in seconds (samples fed so far / 16000) when the event was
produced; ``step`` is the encoder-call index, ``enc_frame`` the 80 ms encoder-frame
index (global, never restarted by a reset), ``frame_t`` its start time.  Processing
latency is not in ``t``.

Encoder input is NeMo's voice-agent layout: a rolling 25-frame feature buffer
(9 pre-encode cache + 16 frames) initialised to LOG_MEL_ZERO (-16.635) that slides by
``chunk_ms/10`` new frames per call (8 for 80 ms, 16 for 160 ms) - "25-frame slice =
9 cache + 16 new" at 160 ms.  At 80 ms the newest 8 frames are look-ahead for the one
emitted encoder frame (att context [70,1]); this needs the shift-1 encoder export.

Frontends (``frontend=``):
  exact      ``StreamingMel``: only NEW frames per chunk, identical to whole-clip
             features (frame i final once sample i*160+200 has arrived)
  nemo_va    replicates NeMo's voice-agent ``CacheFeatureBufferer``: re-featurise the
             last (160 + chunk) samples of a rolling sample buffer and keep the last
             chunk-frames (edge frames see zero padding)
Reset policies (``reset=``): none | decoder (zero the decoder LSTM, last_token=blank,
keep encoder cache and feature stream; parakeet-rs) | full (audio buffer, feature
buffer, encoder caches, decoder state, hypotheses; NeMo voice agent).

``hold_s``: first-output hold.  Events produced before ``hold_s`` of audio has been
fed are queued with their original ``t`` and released on the first feed past it
(``released_t`` set); decoding is never skipped, so the token stream is independent of it.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from .frontend import HOP_LENGTH, N_MELS, SAMPLE_RATE, StreamingMel, offline_mel

BUF_FRAMES = 25
LOG_MEL_ZERO = -16.635     # NeMo voice-agent fill value (log(2**-24) = -16.6355)
ENC_FRAME_S = 0.08


def load_vocab(path: Path) -> list[str]:
    return [ln.rstrip("\n") for ln in Path(path).read_text(encoding="utf-8").splitlines()]


class OrtGraphs:
    """Two onnxruntime CPU sessions (ORT_SEQUENTIAL, inter_op 1).

    Handles both decoder_joint signatures: the parakeet-rs export (with ``target_length``,
    [1,1,1,V] logits) and the community export (no ``target_length``, log-softmax)."""

    def __init__(self, encoder: Path, decoder_joint: Path, threads: int = 1, blank_id: int = 1026):
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        so.log_severity_level = 3
        self.enc = ort.InferenceSession(str(encoder), so, providers=["CPUExecutionProvider"])
        self.dec = ort.InferenceSession(str(decoder_joint), so, providers=["CPUExecutionProvider"])
        self.encoder_path, self.decoder_path = str(encoder), str(decoder_joint)
        self._dec_inputs = {i.name for i in self.dec.get_inputs()}
        self.out_shape = None
        self.blank_id = blank_id
        self._dec_outputs = [o.name for o in self.dec.get_outputs()]

    def encode(self, feats_ct, length, cache):
        out = self.enc.run(None, {
            "audio_signal": feats_ct[None].astype(np.float32),
            "length": np.array([length], dtype=np.int64),
            "cache_last_channel": cache["ch"], "cache_last_time": cache["tm"],
            "cache_last_channel_len": cache["len"]})
        enc, _enc_len, ch, tm, ln = out
        return enc, {"ch": ch, "tm": tm, "len": ln}

    def decode(self, enc_frame, last_token, h, c, hist=()):
        """One joint step.  Two graph families:

        * ``stepwise`` (NeMo ``model.export`` / parakeet-rs layout, has ``target_length``):
          targets=[[last_token]] with the carried LSTM state; the returned state is the next state.
        * ``history`` (community export, no ``target_length``, output axis ``target_plus_sos``): the graph
          PREPENDS an SOS step to ``targets`` on every call, so carrying its state across calls double-counts
          SOS (measured: logits diverge from NeMo after the first token).  It is used exactly by passing the
          committed token history since the last decoder reset with ZERO state and taking the last slot
          (history empty: targets=[[blank]] and slot 0 = the SOS-only prediction).
        """
        if "target_length" in self._dec_inputs:
            feed = {"encoder_outputs": enc_frame, "targets": np.array([[last_token]], dtype=np.int32),
                    "target_length": np.array([1], dtype=np.int32), "input_states_1": h, "input_states_2": c}
            out = dict(zip(self._dec_outputs, self.dec.run(None, feed)))
            o = out["outputs"]
            if self.out_shape is None:
                self.out_shape = tuple(o.shape)
            return o.reshape(-1), out["output_states_1"], out["output_states_2"]
        tg = list(hist) if len(hist) else [self.blank_id]
        feed = {"encoder_outputs": enc_frame, "targets": np.array([tg], dtype=np.int32),
                "input_states_1": np.zeros((1, 1, 640), np.float32), "input_states_2": np.zeros((1, 1, 640), np.float32)}
        out = dict(zip(self._dec_outputs, self.dec.run(None, feed)))
        o = out["outputs"]
        if self.out_shape is None:
            self.out_shape = tuple(o.shape)
        logits = o[0, 0, -1] if len(hist) else o[0, 0, 0]
        return logits, h, c     # state unused in history mode


def new_cache():
    return {"ch": np.zeros((17, 1, 70, 512), np.float32), "tm": np.zeros((17, 1, 512, 8), np.float32),
            "len": np.zeros((1,), np.int64)}


class ParakeetEOUArm:
    def __init__(self, graphs, vocab: list[str], *, chunk_ms: int = 160, reset: str = "none",
                 hold_s: float = 0.0, max_symbols: int = 10, frontend: str = "exact", mel: str = "slaney"):
        assert chunk_ms in (80, 160), chunk_ms
        assert reset in ("none", "decoder", "full")
        assert frontend in ("exact", "nemo_va")
        self.g, self.vocab = graphs, vocab
        self.eou_id = vocab.index("<EOU>")
        self.eob_id = vocab.index("<EOB>") if "<EOB>" in vocab else -1
        self.blank = len(vocab)
        self.chunk_ms, self.chunk_frames = chunk_ms, chunk_ms // 10
        self.chunk_samples = chunk_ms * SAMPLE_RATE // 1000
        self.reset_policy, self.hold_s, self.max_symbols = reset, hold_s, max_symbols
        self.frontend, self.mel_kind = frontend, mel
        self.config = {"chunk_ms": chunk_ms, "reset": reset, "hold_s": hold_s, "max_symbols": max_symbols,
                       "frontend": frontend, "mel": mel, "encoder": getattr(graphs, "encoder_path", None),
                       "decoder_joint": getattr(graphs, "decoder_path", None)}
        self.reset_all()

    # ------------------------------------------------------------------ state
    def _reset_decoder(self):
        self.h = np.zeros((1, 1, 640), np.float32)
        self.c = np.zeros((1, 1, 640), np.float32)
        self.last = self.blank
        self.hist: list[int] = []

    def _reset_stream(self):
        self.mel = StreamingMel(mel=self.mel_kind)
        self.pending = np.zeros((0, N_MELS), np.float32)   # exact: frames not yet consumed
        self.fbuf = np.full((BUF_FRAMES, N_MELS), LOG_MEL_ZERO, np.float32)
        self.sbuf = np.zeros(BUF_FRAMES * HOP_LENGTH, np.float32)  # nemo_va rolling samples
        self.inbox = np.zeros(0, np.float32)                       # nemo_va: samples awaiting a full chunk
        self.cache = new_cache()

    def reset_all(self):
        self._reset_decoder()
        self._reset_stream()
        self.samples = 0
        self.step_no = 0
        self.enc_frames = 0
        self.queue: list[dict] = []
        self.released = self.hold_s <= 0
        self.log: list[dict] = []        # per encoder call: {"step", "tokens"} raw token ids
        self.feat_log: list[np.ndarray] = []
        self.timing = {"encoder_s": [], "decoder_s": []}

    # ------------------------------------------------------------------ feed
    def feed(self, pcm_chunk) -> list[dict]:
        x = np.asarray(pcm_chunk, dtype=np.float32).ravel()
        self.samples += x.size
        events: list[dict] = []
        if self.frontend == "exact":
            self.pending = np.concatenate([self.pending, self.mel.push(x)], axis=0)
            while self.pending.shape[0] >= self.chunk_frames:
                new, self.pending = self.pending[:self.chunk_frames], self.pending[self.chunk_frames:]
                events += self._step(new)
        else:
            self.inbox = np.concatenate([self.inbox, x])
            while self.inbox.size >= self.chunk_samples:
                blk, self.inbox = self.inbox[:self.chunk_samples], self.inbox[self.chunk_samples:]
                events += self._step(self._va_frames(blk))
        return self._gate(events)

    def finish(self) -> list[dict]:
        """End of stream: exact frontend flushes its tail frames (zero right pad) and runs the full chunks."""
        events: list[dict] = []
        if self.frontend == "exact":
            self.pending = np.concatenate([self.pending, self.mel.flush()], axis=0)
            while self.pending.shape[0] >= self.chunk_frames:
                new, self.pending = self.pending[:self.chunk_frames], self.pending[self.chunk_frames:]
                events += self._step(new)
        return self._gate(events, force=True)

    def _va_frames(self, blk: np.ndarray) -> np.ndarray:
        """NeMo CacheFeatureBufferer.update: featurise last (hop + chunk) samples, keep last chunk frames."""
        self.sbuf = np.concatenate([self.sbuf[blk.size:], blk])
        window = self.sbuf[-(HOP_LENGTH + self.chunk_samples):]
        f = offline_mel(window, mel=self.mel_kind)          # floor(len/hop) frames (torch gives +1, trailing one dropped)
        return f[-self.chunk_frames:]

    def _gate(self, events, force=False):
        if self.released:
            return events
        self.queue += events
        if force or self.samples >= self.hold_s * SAMPLE_RATE:
            self.released = True
            out, self.queue = self.queue, []
            now = self.samples / SAMPLE_RATE
            for e in out:
                e["released_t"] = now
            return out
        return []

    def _step(self, new_frames) -> list[dict]:
        self.fbuf = np.concatenate([self.fbuf[self.chunk_frames:], new_frames], axis=0)
        self.feat_log.append(self.fbuf.copy())
        t0 = time.perf_counter()
        enc, self.cache = self.g.encode(self.fbuf.T, BUF_FRAMES, self.cache)
        self.timing["encoder_s"].append(time.perf_counter() - t0)
        step = self.step_no
        self.step_no += 1
        ev: list[dict] = []
        step_toks: list[int] = []
        reset_full = False
        for j in range(enc.shape[2]):
            frame = enc[:, :, j:j + 1].astype(np.float32)
            fi = self.enc_frames
            self.enc_frames += 1
            for _ in range(self.max_symbols):
                t0 = time.perf_counter()
                logits, nh, nc = self.g.decode(frame, self.last, self.h, self.c, self.hist)
                self.timing["decoder_s"].append(time.perf_counter() - t0)
                k = int(np.argmax(logits))
                if k == self.blank:
                    break
                step_toks.append(k)
                self.h, self.c, self.last = nh, nc, k
                self.hist.append(k)     # NeMo: every non-blank (EOU included) advances the decoder
                now = self.samples / SAMPLE_RATE
                base = {"t": now, "step": step, "enc_frame": fi, "frame_t": fi * ENC_FRAME_S, "id": k}
                if k == self.eou_id:
                    ev.append({**base, "kind": "eou"})
                    if self.reset_policy != "none":
                        ev.append({"t": now, "step": step, "enc_frame": fi, "frame_t": fi * ENC_FRAME_S,
                                   "kind": "reset", "policy": self.reset_policy})
                        if self.reset_policy == "decoder":
                            self._reset_decoder()
                        else:
                            reset_full = True
                    continue    # NeMo keeps decoding this frame (up to max_symbols) unless it resets after the step
                if k == self.eob_id:
                    ev.append({**base, "kind": "eob"})
                    if self.reset_policy == "full":     # NeMo voice agent resets on <EOU> or <EOB>
                        ev.append({"t": now, "step": step, "enc_frame": fi, "frame_t": fi * ENC_FRAME_S,
                                   "kind": "reset", "policy": "full"})
                        reset_full = True
                    continue
                ev.append({**base, "kind": "token", "piece": self.vocab[k]})
        self.log.append({"step": step, "tokens": step_toks})
        if reset_full:       # NeMo voice agent resets AFTER the step that contained <EOU>/<EOB>
            self._reset_decoder()
            self._reset_stream()
        return ev


def detok(pieces) -> str:
    return "".join(pieces).replace("▁", " ").strip()
