# M-2 TTS Path-B Observation Runbook

This directory prepares the MiniCPM-o built-in CosyVoice2 probe. It is not a
production integration and every resulting record is `observation-only` under
`/workspace/MEASUREMENT.md`: a valid WAV proves emitted audio bytes only, not
intelligibility, quality, latency, or role fitness.

## Preconditions

1. Do not execute M-2 until the campaign's powered A3/A4 confirmation is terminal.
   The campaign already grants the compute window; this runbook adds no separate
   operator-window request.
2. Create only the isolated detached checkout pinned in `m2_tts_manifest.json`:
   `git clone --filter=blob:none --no-checkout https://github.com/tc-mb/llama.cpp-omni.git /mnt/raid0/llm/llama.cpp-omni-experimental && git -C /mnt/raid0/llm/llama.cpp-omni-experimental checkout --detach 5202b7b2f4d11f50b9f996161e7a2f8b8571b890`.
   This is the `feat/web-demo` commit, not current upstream master.
3. Build and inspect the exact upstream CLI target before proposing an argv file.
   The existing feasibility record reports no supported server target and no
   documented text-only TTS interface. Do not invent flags: capture `--help`
   and the source path that proves each argument.
4. The manifest's `interface_contract` is currently `blocked-unknown-interface`.
   `run` is intentionally impossible until a reviewer records an exact
   source/help-derived argv template, help/source capture hashes, prompt-input
   hash, and argv JSON hash in a separate contract, then pins that contract's
   path and SHA-256 in the manifest.

## Capture

First freeze the built binary and its dynamic-loader view. This hashes all
pinned model components again and refuses a branch checkout, source drift, or
an unavailable executable:

```bash
ROOT=/mnt/raid0/llm/llama.cpp-omni-experimental
RUN=/mnt/raid0/llm/epyc-inference-research/artifacts/minicpm-o-phase1-v8-20260726/m2-tts/runs/$(date -u +%Y%m%dT%H%M%SZ)
python3 m2_tts_observation_runner.py --omni-root "$ROOT" init-run --run-dir "$RUN"
python3 m2_tts_observation_runner.py --omni-root "$ROOT" prepare-runtime-lock --run-dir "$RUN"
```

Write `argv.json` as a JSON string array whose first element is exactly
`$ROOT/build-cpu/bin/llama-omni-cli`; include the output path documented by
the inspected upstream interface. Do not place secrets in argv. Launch only
through the runner:

```bash
python3 m2_tts_observation_runner.py --omni-root "$ROOT" run --run-dir "$RUN" \
  --ack-observation-only
```

The command fails closed unless source is detached at the pinned commit, the
model bundle hashes match, the locked executable is unchanged, all outputs
share the isolated run directory, the child exits zero, and `output.wav` is a
new RIFF/WAVE file with a supported PCM/float format and at least 0.1 seconds
of audio. `capture.json` carries command, source, model, binary/ldd, log, and
audio-header provenance. A failure creates no success record.

## Manual Quality Review

Listen to the emitted WAV and record only a descriptive observation (prompt,
audibility, obvious corruption, language, reviewer). Do not calculate or cite
a quality/latency decision number: no approved TTS quality protocol exists.
Any adoption decision remains blocked on an operator-approved protocol and
the M-3 lineup gate.

An existing candidate WAV can be re-inspected without launching a process:

```bash
python3 m2_tts_observation_runner.py --omni-root "$ROOT" inspect-wav --run-dir "$RUN" \
  --wav "$RUN/output.wav"
```
