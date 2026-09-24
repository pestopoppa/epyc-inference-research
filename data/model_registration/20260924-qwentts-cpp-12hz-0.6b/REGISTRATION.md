# MRG-1 registration — qwentts.cpp / Qwen3-TTS 12Hz 0.6B Talker Q8_0 + 12Hz codec Q8_0

**Task**: INF-41 S-11 (`handoffs/active/multimodal-pipeline.md:303`).
**Registered**: 2026-09-24. **Registry row**: `orchestration/model_registry.yaml`
`server_mode.tts_server` (this repo — the master; the orchestrator's compiled lean
registry carries the same role for launch/data purposes, see that repo's
`orchestration/launch_manifest.yaml` `aux_services.tts`).

This runbook (`docs/protocols/model-registration-runbook.md`, `MRG-1`) is written for
CPU-NUMA-placement `llama-server` roles with speculative decoding, `-np` slot sweeps and
a context-length axis. `qwentts.cpp`'s `tts-server` is a single-instance, GPU-resident,
non-context-scaling aux service (backend `tts` in `AUX_SERVICES`, not a
`role_launch_meta`/`stack_topology.yaml` NUMA role). Steps 3–7 are therefore genuinely
**N/A by construction**, not skipped — see the reasons column. Steps 0, 1, 2, 8 and 9
are executed.

## Step table

| # | Step | Verdict | Artifact | Note |
|---|---|---|---|---|
| 0 | Identity | PASS | `identity.txt` | talker + codec GGUF pair, path/size/sha verified against the LIVE launcher config, not the stale task-text pair |
| 1 | Capacity arithmetic | PASS (simplified) | `capacity.txt` | fixed VRAM-resident footprint, no `-c`-scaling KV axis |
| 2 | Production recipe | PASS, spec-dec N/A | registry `acceleration: {type: none}` | R1-CHK `type: none` carve-out applies |
| 3 | Anchor gate | N/A — reason: no tok/s-shaped metric exists for this role; evidence is round-trip WER + RTF, not a `P-BENCH-*` decode rate | — | — |
| 4 | Placement | N/A — reason: GPU-resident weights, no host NUMA placement axis | — | — |
| 5 | Shape × concurrency | N/A — reason: single-instance aux service, not a multi-slot `llama-server` role | — | — |
| 6 | Context curve | N/A — reason: no `-c`-shaped context axis for a talker+codec TTS pair | — | — |
| 7 | Slot width | N/A — reason: single request in flight under the current aux-service wiring | — | — |
| 8 | Co-residency | PASS (informal) | `coresidency.txt` | measured co-resident with `architect_general` + `worker_vision` + whisper on the MI210 throughout 2026-09; no isolated idle-vs-active A/B run — this is the model that made the capacity gate's blind spot real |
| 9 | Registration record | this directory | — | — |

## Step 0 — Identity

- `path`: `/mnt/raid0/llm/models/Qwen3-TTS-qwentts/qwen-talker-0.6b-base-Q8_0.gguf` (992,615,488 B),
  sha256 `d54dbaf10591421fa764ed630d764efa717ae40cd959bd48c66d4eb1af226426` (see `identity.txt`)
- `codec_path`: `/mnt/raid0/llm/models/Qwen3-TTS-qwentts/qwen-tokenizer-12hz-Q8_0.gguf` (291,150,624 B),
  sha256 `1883beeed99348fc35e23dd225e9082f93f6f8c109330a33d935baa8acdbfd94` (see `identity.txt`)
- combined: 1,283,766,112 B = 1.196 GiB, matches the file's own 2026-07-31 "1.19 GB" banner
- quant: Q8_0 (both files)
- **Path correction**: `multimodal-pipeline.md` S-11's task text names
  `/mnt/raid0/llm/models/Qwen3-TTS-12Hz-0.6B-{Talker-Q4_K_M,CodePredictor-Q8_0}.gguf`. Those files
  exist (484,219,712 B / 420,157,632 B) but are leftovers of the abandoned, unrebuildable Path A
  home-rolled llama.cpp port (`llama-tts-qwen3`; source lost). They are **not** what `qwentts.cpp`
  serves. The path above is read directly from the live `epyc-orchestrator`
  `orchestration/launch_manifest.yaml` `aux_services.tts` entry, which is what `:9002` has
  actually run since 2026-08-02 (W4).
- kernel pin: `qwentts.cpp` @ `production-speech-v1`, commit `2c1b5182e7e9f1acaa04405ff21747d8a7acf4d5`,
  ggml submodule `b86f660238dcc1a83b7cbf5a72d355a965de9245` (ggml 0.17.0), binary
  `/mnt/raid0/llm/qwentts.cpp/build/tts-server`, sha256
  `369fc2f1de88e41f4459e1f56c0e962035e20984acf0ea2d4678f602232ff654`.
  Source: `/workspace/artifacts/operator/ratify_speech_kernel_freeze_20260731.json`.
- **Pin correction (S-13 motivation)**: S-11's task text records commit `abab6b3b` / ggml fork
  `c044c6f0` / binary md5 `5b858d75614dfd2f696071212ae8f2e4`. Verified 2026-09-24:
  `git -C /mnt/raid0/llm/qwentts.cpp merge-base --is-ancestor abab6b3b 2c1b5182e` returns true —
  `abab6b3b` is a direct ancestor of `2c1b5182e`, one commit back (`git log --oneline abab6b3b..2c1b5182e`
  shows exactly the one freeze commit, "freeze: pin ggml submodule carrying the gfx90a GPU patches",
  2026-07-31 16:11:39Z). That commit recorded ggml submodule state that had been **uncommitted** dirty
  state at `abab6b3b`'s time — the measured binary could not be rebuilt from `abab6b3b` alone. The
  on-disk binary's sha256 matches the ratification exactly; its md5
  (`b976c78cc5ca32668ee4f76b2ce5c743`) matches neither pin text's stated md5 anywhere on this host.
  The live `launch_manifest.yaml` `tts` entry's own `model_label` already reads "ggml 0.17.0".
  **Verdict**: the ratified freeze (`2c1b5182e` / ggml 0.17.0) is ground truth and is what this
  registration records; S-11's pin text was stale (written before the freeze commit landed).

## Step 1 — Capacity arithmetic

No `-c`-shaped KV axis: `qwentts.cpp`'s talker + codec pair hold a fixed resident footprint,
measured live rather than derived from a per-token KV formula. Measured 2026-09-22 from
`/sys/class/kfd/kfd/proc/<pid>/vram_57300`, sampled **during** the running serving stack:
`2,815,950,848 B = 2.62 GiB`. Source: `artifacts/operator/lineup-change-20260922.md`
("Live per-process VRAM ... tts-server (Qwen3-TTS) ... 2.62 GiB"). This is the figure the
capacity-gate visibility fix (INF-41, prepared on noninf/inf41-speech, NOT landed) will consume — see `epyc-orchestrator`
`orchestration/launch_manifest.yaml` `aux_services.tts.vram_gib` and
`scripts/server/stack_manifest.py` `serving_shape_capacity_report()`.

## Step 2 — Production recipe

`acceleration: {type: none}` in the registry row. No speculative decoding, no draft model.
MRG-1 R1-CHK's `type: none` carve-out applies (§ Step 2, trap 3 in the runbook): a `type: none`
role's absence of speculation arguments IS the production recipe, and every figure is already
labelled accordingly (no bare tok/s headline is claimed anywhere in the registry row).

## Step 8 — Co-residency

Measured co-resident with `architect_general` (Qwen3.8-27B, `:8083`) and `worker_vision`
(Qwen3-VL-30B-A3B, `:8086`) on the same MI210 throughout 2026-09, including the 2026-09-22
live sample recorded in `artifacts/operator/lineup-change-20260922.md`. No isolated
idle-vs-active co-residency A/B run has been performed for this model specifically (unlike
the GPU-shadow-lane program's `gpuoverlap.sh`/`gpuoverlap2.sh` proxies for CPU roles) — the
"measurement" here is the live production stack itself, which is why this row is the reason
the capacity gate's 4.68 GiB blind spot (this model's 2.62 GiB share, plus whisper's 2.06 GiB)
was discovered in the first place.

## Claim grammar

`Qwen3-TTS-12Hz-0.6B Q8_0 (talker+codec) qwentts.cpp GPU round-trip WER 1.49%, TTFA 37.8ms,
RTF 0.169 (5.9x realtime), MI210 [S-6/S-6a, multimodal-pipeline.md, n unspecified,
2026-07-31, observation-grade — not a ratified P-BENCH-* protocol]`

## Re-registration triggers (§9.4)

Re-run on: a change to the talker or codec GGUF bytes; a `qwentts.cpp`/ggml-fork commit change
(pin drift — see the S-13 guard, `scripts/utils/verify_qwentts_pin_isolation.sh`); a host
topology change (NPS reboot) if the GPU host-lane thread pinning is ever added for this
service; or a change to the `AUX_SERVICES.tts` launch recipe.
