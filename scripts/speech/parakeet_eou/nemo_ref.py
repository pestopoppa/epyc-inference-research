"""CS-46 (e): NeMo CPU streaming reference for nvidia/parakeet_realtime_eou_120m-v1.

Runs in the ISOLATED CPU venv (NeMo >= 2.5.3, torch CPU).  It drives NeMo's own
voice-agent streaming ASR service (``nemo.agents.voice_agent...NemoStreamingASRService``,
shipped inside the NeMo wheel) unmodified, on CPU, one fixed-size audio chunk per call.
The pipecat package that the service's package ``__init__`` imports is not needed by
the ASR service itself, so the parent packages are stubbed (namespace modules); the
service and ``CacheFeatureBufferer`` code is NeMo's.

Per clip it writes ``<id>.json`` (raw per-step token ids, EOU frames, text) and
``<id>.mel.npy`` (whole-clip offline features of NeMo's own ``model.preprocessor``,
eval mode, [T,128]) plus ``<id>.va_feats.npy`` (the 25-frame feature buffer NeMo's
service fed the encoder at every step, [steps,25,128]).

Reset modes (--reset): ``full`` = the service as shipped (on <EOU>/<EOB> it resets audio
buffer, encoder caches and hypotheses); ``none`` = reset_state patched to a no-op
(tokens run on through EOU), the reference for the ``none`` variant.

usage: python -I nemo_ref.py --nemo FILE.nemo --clips clips.jsonl --out DIR --chunk-ms 80|160 --reset full|none
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import types
from pathlib import Path

import numpy as np


def stub_pipecat_parents():
    import nemo
    base = Path(nemo.__file__).parent / "agents" / "voice_agent" / "pipecat" / "services" / "nemo"
    chain = ["nemo.agents", "nemo.agents.voice_agent", "nemo.agents.voice_agent.pipecat",
             "nemo.agents.voice_agent.pipecat.services", "nemo.agents.voice_agent.pipecat.services.nemo"]
    for name in chain:
        m = types.ModuleType(name)
        parts = name.split(".")[1:]
        m.__path__ = [str(Path(nemo.__file__).parent.joinpath(*parts))]
        sys.modules[name] = m
    assert base.exists()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nemo", required=True)
    ap.add_argument("--clips", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--chunk-ms", type=int, required=True, choices=(80, 160))
    ap.add_argument("--reset", default="full", choices=("full", "none"))
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--frontend-mode", default="shipped", choices=("shipped", "eval"),
                    help="shipped: the service as NeMo ships it (its CacheFeatureBufferer builds a fresh preprocessor "
                         "that stays in TRAINING mode, so dither=1e-5 noise is added; seeded per clip). "
                         "eval: preprocessor.eval() (no dither) = deterministic reference")
    a = ap.parse_args(argv)
    import torch
    torch.set_num_threads(a.threads)
    import soundfile as sf
    stub_pipecat_parents()
    import nemo
    from nemo.agents.voice_agent.pipecat.services.nemo.streaming_asr import NemoStreamingASRService

    class Rec(NemoStreamingASRService):
        def __init__(self, *args, no_reset=False, **kw):
            self._no_reset = no_reset
            self.rec_tokens, self.rec_feats = [], []
            super().__init__(*args, **kw)

        def _get_tokens_and_probs_from_alignments(self, alignments):
            toks, probs = super()._get_tokens_and_probs_from_alignments(alignments)
            self.rec_tokens.append([int(t) for t in toks])
            return toks, probs

        def reset_state(self, stream_id="default"):
            if self._no_reset:
                return
            super().reset_state(stream_id)

        def hard_reset(self):
            NemoStreamingASRService.reset_state(self)

    svc = Rec(model=a.nemo, device="cpu", chunk_size_in_secs=a.chunk_ms / 1000.0, no_reset=(a.reset == "none"))
    pre_training_as_shipped = bool(svc._audio_buffer.preprocessor.training)
    if a.frontend_mode == "eval":
        svc._audio_buffer.preprocessor.eval()
    a.out.mkdir(parents=True, exist_ok=True)
    cfgecho = {"nemo_version": nemo.__version__, "torch": torch.__version__, "args": {k: str(v) for k, v in vars(a).items()},
               "streaming_cfg": str(svc.asr_model.encoder.streaming_cfg), "att_context_size": list(svc.att_context_size),
               "buffer_size_in_secs": svc.buffer_size_in_secs, "tokens_per_frame": svc.tokens_per_frame,
               "preprocessor_dither_training_only": float(svc.asr_model.preprocessor.featurizer.dither),
               "preprocessor_normalize": str(svc.asr_model.preprocessor.featurizer.normalize),
               "service_preprocessor_training_as_shipped": pre_training_as_shipped,
               "service_preprocessor_dither": float(svc._audio_buffer.preprocessor.featurizer.dither),
               "frontend_mode": a.frontend_mode}
    (a.out / "config_echo.json").write_text(json.dumps(cfgecho, indent=1))
    eou_id, eob_id, blank = svc.tokenizer.vocab.index("<EOU>"), svc.tokenizer.vocab.index("<EOB>"), svc.blank_id
    clips = [json.loads(l) for l in a.clips.read_text().splitlines() if l.strip()]
    chunk = int(16000 * a.chunk_ms / 1000)
    for c in clips:
        x, sr = sf.read(c["path"], dtype="float32", always_2d=True)
        assert sr == 16000
        x = x[:, 0]
        pcm16 = np.clip(np.round(x * 32768.0), -32768, 32767).astype("<i2")
        pad = (-len(pcm16)) % chunk
        pcm16p = np.concatenate([pcm16, np.zeros(pad, "<i2")])
        torch.manual_seed(0)
        svc.hard_reset()
        svc.rec_tokens.clear()
        steps, feats = [], []
        t0 = time.perf_counter()
        for i in range(0, len(pcm16p), chunk):
            svc.rec_tokens.clear()
            r = svc.transcribe(pcm16p[i:i + chunk].tobytes())
            toks = svc.rec_tokens[-1] if svc.rec_tokens else []
            steps.append({"call": i // chunk, "end_sample": i + chunk, "tokens": toks, "text": r.text, "is_final": r.is_final})
            if a.reset == "none":
                feats.append(svc._audio_buffer.get_feature_buffer().numpy().T.copy())
        wall = time.perf_counter() - t0
        flat = [t for s in steps for t in s["tokens"]]
        with torch.no_grad():
            f, fl = svc.asr_model.preprocessor(input_signal=torch.from_numpy(pcm16.astype(np.float32) / 32768.0)[None],
                                                length=torch.tensor([len(pcm16)]))
        mel = f[0, :, : int(fl[0])].numpy().T.copy()
        eou_calls = [s["call"] for s in steps if eou_id in s["tokens"]]
        out = {"clip_id": c["clip_id"], "pcm_i16_sha256": hashlib.sha256(pcm16.tobytes()).hexdigest(),
               "chunk_ms": a.chunk_ms, "reset": a.reset, "n_samples": int(len(pcm16)), "tokens": flat,
               "eou_calls": eou_calls, "eou_frames": eou_calls, "steps": steps, "wall_s": wall,
               "eou_id": eou_id, "eob_id": eob_id, "blank_id": blank, "mel_frames": int(mel.shape[0])}
        (a.out / f"{c['clip_id']}.json").write_text(json.dumps(out))
        np.save(a.out / f"{c['clip_id']}.mel.npy", mel)
        if feats:
            np.save(a.out / f"{c['clip_id']}.va_feats.npy", np.stack(feats))
        print(c["clip_id"], "tokens", len(flat), "eou_calls", eou_calls, f"{wall:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
