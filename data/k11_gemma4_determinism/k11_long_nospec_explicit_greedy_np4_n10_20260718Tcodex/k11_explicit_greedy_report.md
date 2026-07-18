# K11 No-Spec Explicit-Greedy Diagnostic - 2026-07-18

Purpose: test whether the K11 long exact-stop failures are caused by the
historical request sampler shape or ROCm backend top-k sampling.

Setup:

- Server: experimental-v7 HIP `llama-server`
- Model: `gemma-4-26B-A4B-it-ORIG-Q4_K_M.gguf`
- Speculation: `--spec-type none`
- Slots: `4`
- Request sampler mode: `explicit-greedy`
- Request payload changes: `samplers=["temperature"]`, `temperature=0`,
  `top_k=0`, `top_p=1`, `min_p=0`, `backend_sampling=false`
- Prompt: exactly 200 `benchmark` words
- Runs: 10 fresh sequential servers

Result:

| Metric | Value |
|---|---:|
| Runs | 10 |
| Unique output hashes | 3 |
| Task passes | 4/10 |
| 200-word outputs | 4 |
| 289-word outputs | 2 |
| 512-word cap hits | 4 |
| Non-`benchmark` words | 0 |

Interpretation:

- Explicit CPU-side greedy sampling did not fix no-spec stop/count divergence.
- The ROCm `TOP_K` backend-sampler warning is unlikely to be the root cause for
  target generation semantics on this gate.
- Because every output used only the requested word, this diagnostic points at
  stop/count compliance or server/model termination behavior rather than token
  corruption.
- Do not promote multi-slot Gemma4 GPU worker serving from K11. The next useful
  diagnostic should use structural termination (`stop` string or JSON schema)
  to separate exact count-following from deterministic token choice.

Cleanup:

- Runner verified all temporary server PIDs exited.
- Post-run checks found no `llama-server`, K11 runner, AutoPilot, or KFD process.
