# Speech kernel freeze 2026-07-31 — STT WER evidence

Preserved 2026-08-04 from `/mnt/raid0/llm/tmp/stt_wer_results.json`, which is an **ephemeral scratch
root** (the first entry of `autokernel/storage.py::EPHEMERAL_ROOTS`) and was tracked in no
repository. It is the primary record behind
`epyc-root/artifacts/operator/ratify_speech_kernel_freeze_20260731.json`, so losing it would have
left a ratified receipt with no recoverable evidence — the exact failure the 2026-08-02
evidence-durability ruling exists to prevent.

Bytes are unchanged. Digest in `SHA256SUMS`:
`266333e9a2b1b12c17f9b9b27c168ea450f87dad1a3217029e061ef41d1b1e74`.

## Arms, verbatim from the file

| arm | WER % | errors / ref words |
|---|---|---|
| `faster-whisper large-v3-turbo int8 CPU 48t` | **2.35** | 44 / 1870 |
| `Qwen3-ASR-1.7B Q8_0 MI210 GPU` | 72.14 | — / 1870 |
| `Qwen3-ASR-1.7B Q8_0 + bf16 projector, MI210` | 29.36 | — / 1870 |
| `whisper.cpp large-v3-turbo f16 MI210 GPU` | **3.37** | 63 / 1870 |
| `whisper.cpp large-v3-turbo f16 MI210 GPU beam5` | 3.26 | 61 / 1870 |
| `whisper.cpp large-v3 f16 MI210 GPU` | 3.32 | 62 / 1870 |

n = 100 utterances per arm.

## Why this directory exists

The receipt records `whisper_cpp.measurements_anchored.wer_pct = 2.35`. That is the
**faster-whisper large-v3-turbo int8 CPU 48t** arm — CTranslate2, a different engine on a different
runtime, on CPU rather than the MI210. The production configuration is `whisper.cpp
large-v3-turbo f16 MI210`, whose WER in the same file is **3.37 %**.

The receipt therefore anchors the whisper.cpp kernel to a number whisper.cpp never produced, about
one percentage point in the flattering direction. The consequence is forward-looking rather than
historical: every future `whisper_stt` non-inferiority comparison inherits that denominator, so a
candidate that genuinely matched production's real 3.37 % would read as a ~1 pp **regression**
against a baseline it never had — and a candidate that was genuinely 1 pp worse would read as
parity.

Correction is a **superseding receipt**, never an in-place edit: `MEASUREMENT.md:174-175` forbids
destroying primary records, and `measurement/protocols/bench-cpu.md:163-168` is the in-corpus
precedent for the supersession form. Staged as
`epyc-root/artifacts/operator/ratify_speech_wer_correction_20260804.json`, unsigned — only a human
may amend a ratified receipt.

## A second gap in the same receipt

`qwentts_cpp.measurements_anchored.roundtrip_wer_pct = 1.49` names **no STT instrument**. A
round-trip WER measures synthesized audio *through a recognizer*, so the number is a property of the
pair, not of the TTS kernel alone — and the recognizer used here is unrecorded. `P-TTS-2`, ratifying
as part of Annex S, requires `stt_instrument=<binary_sha256[:12]>/<model_sha256[:12]>` in its
grammar for exactly this reason. The existing figure cannot satisfy that grammar and should be
treated as provenance, not as a baseline, until it is re-measured with the instrument named.
