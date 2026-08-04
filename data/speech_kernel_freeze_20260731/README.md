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

## Follow-up audit, 2026-08-04: the latency figure is CORRECT

The supersession receipt (`ratify_speech_wer_correction_20260804.json`, RATIFIED
2026-08-04T15:48:15Z) left one operator action open: *"decide whether
`latency_s_11s_clip` warrants the same attribution audit — it was not checked."* It is
checkable from this file, so it was checked rather than left as a decision.

The receipt records `whisper_cpp.measurements_anchored.latency_s_11s_clip = 0.21` for an
11-second clip. Dividing 11 s by each arm's `xrt_overall`:

| arm | xrt_overall | implied latency for 11 s |
|---|---|---|
| `faster-whisper large-v3-turbo int8 CPU 48t` | 1.58 | 6.962 s |
| `Qwen3-ASR-1.7B Q8_0 MI210 GPU` | 4.02 | 2.736 s |
| `Qwen3-ASR-1.7B Q8_0 + bf16 projector, MI210` | 17.55 | 0.627 s |
| **`whisper.cpp large-v3-turbo f16 MI210 GPU`** | **51.86** | **0.212 s** ← recorded 0.21 |
| `whisper.cpp large-v3-turbo f16 MI210 GPU beam5` | 40.49 | 0.272 s |
| `whisper.cpp large-v3 f16 MI210 GPU` | 24.02 | 0.458 s |

The match is 2.1 ms and the nearest competing arm is 3× away. **The latency figure came
from the production whisper.cpp arm. Only the WER came from faster-whisper.**

**Why that conclusion is worth more than closing one checkbox.** Both figures were
transcribed out of this same file into adjacent lines of the same receipt, and one of them
is right. So the defect was a **single copy error on one line**, not systematic
mis-sourcing of the speech freeze's measurements — which means the receipt's other anchored
values do not need re-auditing on suspicion. A correction whose scope nobody bounded is how
a one-line error turns into a re-audit of everything it sat next to.

Still open, and not fixable by correction: `qwentts_cpp.measurements_anchored.
roundtrip_wer_pct = 1.49` names no STT instrument (see above). `P-TTS-2`, ratified in Annex
S on 2026-08-03, now requires `stt_instrument=<binary_sha256[:12]>/<model_sha256[:12]>` in
its grammar, so that figure can no longer satisfy the protocol governing it. It gets fixed
by re-measuring, which needs the same quiet host the first AutoKernel campaign is waiting on.
