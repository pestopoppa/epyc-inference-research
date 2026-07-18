# K11 No-Spec Explicit-Greedy Stop Diagnostic - 2026-07-18

Purpose: test whether a structural `stop` string closes the K11 long
stop/count divergence after CPU-side greedy request controls.

Setup:

- Server: experimental-v7 HIP `llama-server`
- Model: `gemma-4-26B-A4B-it-ORIG-Q4_K_M.gguf`
- Speculation: `--spec-type none`
- Slots: `4`
- Request sampler mode: `explicit-greedy`
- Stop strings: `["END"]`
- Prompt: exactly 200 `benchmark` words, then `END`
- Runs: 10 fresh sequential servers

Result:

| Metric | Value |
|---|---:|
| Runs | 10 |
| Unique output hashes | 1 |
| Determinism | pass |
| Task passes | 0/10 |
| Observed words | 512 in every run |
| Non-`benchmark` words | 0 |
| Stop marker emitted | no |

Interpretation:

- Adding the `stop` payload made the output deterministic in this no-spec
  control, but did not solve task compliance.
- The model never emitted the `END` marker, so the server-side stop mechanism
  had nothing to intercept.
- K11 remains open. The next useful diagnostic is a schema/grammar or prompt
  shape where the termination marker is structurally unavoidable, rather than a
  natural-language instruction to emit `END`.

Cleanup:

- Runner verified all temporary server PIDs exited.
- Post-run checks found no K11 runner, Gemma4 `llama-server`, AutoPilot, or KFD
  process.
