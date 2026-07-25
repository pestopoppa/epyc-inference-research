# Laguna IQ2 MI210 K/V and Flash-Attention Sweep

**Date:** 2026-07-25
**Status:** Observation only; not a promotion gate or global optimum claim
**Raw artifact:** `data/gpu-mi210/laguna-iq2-kv-sweep-exact-tip/run-20260725T125201Z`

## Fixed identity

- Candidate: `67a433bf45a8a091d83b4ea0b32ff0735fd51800`
- Server version: `10107`
- Server SHA256: `094b395244d71f0d30f82999e53d261f9d4daeea0b651b3e95b51cb6712888ac`
- Target: `Laguna-S-2.1-UD-IQ2_M.gguf`
- Target SHA256: `1a0d44795f71044de1a9671bf70def4655f4ab7294b002263dfc8046820bfd2c`
- Captured harness SHA256: `e366b05dc6fdfc6d0032aaa0449c7e9f497b48fe677098423d269bc6b76a66bb`
- Pre/post source, server, shared-library, model, harness, and execution-binding witnesses matched.

## Protocol

The target was fully resident on MI210 GPU 0 at context 4096. Three target-only
configurations ran as a cyclically counterbalanced 15-server matrix with five
fresh-server replicates per cell and three fixed semantic prompts per replicate.
Every replicate required normal completion, semantic and sanity checks, exact
target residency, dead-process proof, a clean KFD/process postflight, and settled
VRAM.

## Results

| Cell | K/V cache | Flash attention | Decode median | Decode MAD | Prompt median | Result |
|---|---|---:|---:|---:|---:|---|
| A | q8_0 / q8_0 | on | 33.992845 t/s | 0.082450 | 227.249018 t/s | 5/5 pass |
| B | f16 / f16 | on | 35.490117 t/s | 0.283348 | 230.008124 t/s | 5/5 pass |
| C | f16 / f16 | off | 33.782293 t/s | 0.147407 | 37.475456 t/s | 5/5 pass |

Cell B is the bounded best observed configuration. Relative to A, its median
decode ratio is `1.0440466824` (`+4.404668%`) and its prompt-throughput ratio is
`1.0121413353` (`+1.214134%`).

Earlier preserved attempts exposed unsupported q8 V-cache without flash
attention, output instability in the supported flash-off cell, and delayed ROCm
VRAM release after process death. The final harness rejects the unsupported
configuration, retains semantic failures, and polls VRAM settlement while
failing immediately on any process or KFD contamination.

## Artifact hashes

- `plan.json`: `b3efbe05766ccc1eaf48e33970c25fd31b53c19719323e771e7e7fe464a8e37f`
- `summary.json`: `50412804350c87c0b0a3c0f7f84a20944437d913419b115a9ff5d3c4fd8c789b`
- `identities.json`: `bf3d37f1709bfe33e36a1b846ac148b21fe5d8e2684289df7641bf6c6f745fd6`
- `harness_source.py`: `e366b05dc6fdfc6d0032aaa0449c7e9f497b48fe677098423d269bc6b76a66bb`

An independent read-only audit accepted all 15 per-cell semantic, residency,
cleanup, VRAM, identity, calculation, and harness-snapshot witnesses.
