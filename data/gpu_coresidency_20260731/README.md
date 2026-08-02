# GPU co-residency — LLM + vision + STT + TTS on one MI210

Measurement evidence migrated out of scratch on **2026-08-02**. The master registry
cited these artifacts at their `/mnt/raid0/llm/tmp/` paths, so a routine sweep of that
directory would have left ratified, production-affecting claims with nothing behind
them. Copied byte-for-byte (sha256-verified both ends); the scratch originals were
left in place.

| | |
|---|---|
| scratch origin | `/mnt/raid0/llm/tmp/gpu_coresidency` |
| scratch origin | `/mnt/raid0/llm/tmp/gpu_coresidency_results.json` |
| measured (file mtimes, UTC) | 2026-07-31 14:52 .. 2026-07-31 15:03 |
| migrated | 2026-08-02 |
| carried | 20 files, 315,547 bytes |

## What this measured

Concurrent-load probe: an LLM, a vision model, whisper.cpp STT and qwentts.cpp TTS held resident on the same MI210 while decode throughput was sampled. `gpu_coresidency_results.json` is the compiled result; `qwen27b.log` is the arm the registry quotes by line number.

## Registry claims this backs

`orchestration/model_registry.yaml` — these citations resolve to this directory.
The YAML key path is the stable reference; line numbers are as of 2026-08-02.

- **L1166** &nbsp;`server_mode.architect_general.throughput`
  > # data/gpu_coresidency_20260731/qwen27b.log. Never a headline.
- **L1168** &nbsp;`server_mode.architect_general.throughput`
  > # MI210, fully reversible (data/gpu_coresidency_20260731/gpu_coresidency_results.json).
- **L2136** &nbsp;`roles.architect_general.performance.baseline_tps`
  > # data/gpu_coresidency_20260731/qwen27b.log.
- **L9439** &nbsp;`roles.qwen36_27b_mtp_q8_local.production_throughput.baseline_tps`
  > # data/gpu_coresidency_20260731/qwen27b.log). Never a headline.
- **L9442** &nbsp;`roles.qwen36_27b_mtp_q8_local.production_throughput.attest`
  > attest: "published stack measurement record §01, 2026-07-31; co-residency data/gpu_coresidency_20260731/gpu_coresidency_results.json"
- **L9962** &nbsp;`routing_hints.use`
  > # (data/gpu_coresidency_20260731/gpu_coresidency_results.json). The throughput priors below are

## Integrity

`SHA256SUMS` lists every carried file, hashed after the copy and compared against the
scratch original. Verify with:

```bash
cd /mnt/raid0/llm/epyc-inference-research && sha256sum -c data/gpu_coresidency_20260731/SHA256SUMS
```

## Not carried (hash-only)

Too large for this repository. Recorded here so the artifact stays identifiable and
the hash stays checkable against the scratch original:

| file | bytes | sha256 |
|---|---:|---|
| `/mnt/raid0/llm/tmp/gpu_coresidency/tts_stream.wav` | 16,619,200 | `c3656cc7eefcd5930ed32f957f69c449cf372348838130d623fb747981c7bc09` |

`tts_stream.wav` is the raw audio the TTS loop emitted while the four models shared
the GPU — a byproduct of generating load, not a measurement result. Every number the
registry cites comes from the logs and `gpu_coresidency_results.json`, which are
carried here in full. Re-verify the blob against its hash above while the scratch
original still exists; once it is swept the hash is a record, not a check.

