# MRG-1 registration — whisper.cpp large-v3-turbo f16 (MI210, HIP/gfx90a)

**Task**: INF-41 S-12 (`handoffs/active/multimodal-pipeline.md:304`).
**Registered**: 2026-09-24. **Registry row**: `orchestration/model_registry.yaml`
`server_mode.voice_server` (this repo — the master). Retires the CPU faster-whisper
description that row carried until this registration; see `retired_cpu_path` in the row
for the historical record (kept, not deleted, per registry convention for role
retirements that are not model-file deletions).

Same scope note as the qwentts.cpp registration
(`../20260924-qwentts-cpp-12hz-0.6b/REGISTRATION.md`): this runbook targets CPU-NUMA
`llama-server` roles; `whisper-server` is a single-instance, GPU-resident, non-context-scaling
aux service. Steps 3–7 are N/A by construction.

## Step table

| # | Step | Verdict | Artifact | Note |
|---|---|---|---|---|
| 0 | Identity | PASS | `identity.txt` | large-v3-turbo f16, path/sha256 verified |
| 1 | Capacity arithmetic | PASS (simplified) | `capacity.txt` | fixed VRAM-resident footprint; whisper's own cross-attention cache is bounded by its fixed 30s mel window, not `-c` |
| 2 | Production recipe | PASS, spec-dec N/A | registry `acceleration: {type: none}` | R1-CHK `type: none` carve-out applies |
| 3 | Anchor gate | N/A — reason: no tok/s-shaped metric; evidence is WER + wall-clock latency under the ad hoc `stt_wer.py` harness (observation-grade, not a ratified `P-BENCH-*` id) | — | — |
| 4 | Placement | N/A — reason: GPU-resident weights, no host NUMA placement axis | — | — |
| 5 | Shape × concurrency | N/A — reason: single-instance aux service, not a multi-slot `llama-server` role | — | — |
| 6 | Context curve | N/A — reason: no `-c`-shaped context axis; fixed 30s mel window per request | — | — |
| 7 | Slot width | N/A — reason: single request in flight under the current aux-service wiring | — | — |
| 8 | Co-residency | PASS (informal) | `coresidency.txt` | measured co-resident with `architect_general` + `worker_vision` on the MI210 throughout 2026-09; this is the other half of the capacity gate's 4.68 GiB blind spot |
| 9 | Registration record | this directory | — | — |

## Step 0 — Identity

- `path`: `/mnt/raid0/llm/models/whisper-ggml/ggml-large-v3-turbo.bin` (1,624,555,275 B)
- sha256: `1fc70f774d38eb169993ac391eea357ef47c88757ef72ee5943879b7e8e2bc69`
- quant: f16
- kernel pin: `whisper.cpp` @ `production-speech-v1`, commit `b307379226d93d9c5ed790d7cea0626613c0ef4b`,
  ggml 0.18.0, binary `/mnt/raid0/llm/whisper.cpp/build/bin/whisper-server`, sha256
  `82aa8b569b7c8ee031f7a8bba6b21425b760654ea05e1d99991067d5d9bd9c7b`.
  Source: `/workspace/artifacts/operator/ratify_speech_kernel_freeze_20260731.json`. No pin
  discrepancy found for whisper.cpp (unlike qwentts.cpp/S-11 — see the sibling registration).

## Step 1 — Capacity arithmetic

No `-c`-shaped KV axis at the registry level: whisper.cpp pads every request to a fixed 30s
mel window internally, so the resident footprint is measured live rather than derived from a
per-token formula. Measured 2026-09-22 from `/sys/class/kfd/kfd/proc/<pid>/vram_57300`, sampled
**during** the running serving stack: `2,210,533,376 B = 2.06 GiB`. Source:
`artifacts/operator/lineup-change-20260922.md` ("Live per-process VRAM ... whisper-server ...
2.06 GiB"). This is the figure the capacity-gate visibility fix (S-13) consumes — see
`epyc-orchestrator` `orchestration/launch_manifest.yaml` `aux_services.whisper.vram_gib` and
`scripts/server/stack_manifest.py` `serving_shape_capacity_report()`.

## Step 2 — Production recipe

`acceleration: {type: none}` in the registry row (int8/VAD fields from the retired CPU path
are preserved under `retired_cpu_path.acceleration_was`, not carried into the live row). No
speculative decoding. R1-CHK's `type: none` carve-out applies.

## Step 8 — Co-residency

Measured co-resident with `architect_general` and `worker_vision` on the same MI210 throughout
2026-09, including the 2026-09-22 live sample in `artifacts/operator/lineup-change-20260922.md`.
No isolated idle-vs-active A/B run has been performed for this model specifically — same caveat
as the qwentts.cpp sibling registration.

## Claim grammar

`whisper-large-v3-turbo f16 whisper.cpp GPU WER 2.35%, wall median 0.124s / max 0.218s, encode
3751ms->110ms, MI210 [S-10, multimodal-pipeline.md, n=100 (LibriSpeech test-clean), 2026-07-31,
decision-grade for the WER figure per that section's own protocol note]`

## Re-registration triggers (§9.4)

Re-run on: a change to the GGUF bytes; a `whisper.cpp` commit change (pin drift —
`scripts/session/verify_speech_kernels.sh` already enforces branch + binary sha256); a host
topology change; or a change to the `AUX_SERVICES.whisper` launch recipe.
