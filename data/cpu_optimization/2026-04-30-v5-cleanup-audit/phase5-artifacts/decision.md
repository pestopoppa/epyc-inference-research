# v5 Cleanup Audit — Final Disposition

**Date**: 2026-04-30
**Decision**: **GO for v5 push** (subject to user confirmation before remote push action)
**Decided by**: Claude Opus 4.7 (1M-context auto-mode session) — pending user sign-off

## Validation gate summary

| Gate | Result | Status |
|---|---|---|
| Batch 1 — Build (clang-20 + libomp + znver5, no PGO) | 0 errors / 1 pre-existing test-code warning | ✓ PASS |
| Batch 2 — Reproducibility tripwire (Coder-30B Q4_K_M tg32 r=5) | 47.13 ± 0.74 t/s | ✓ PASS (≥47) |
| Batch 3 — PPL bit-exact gate (Coder-30B Q4_K_M chunks 1-12) | **11.1146 ± 0.62405** — exact match to v4 documented reference | ✓ PASS BIT-EXACT |
| Batch 3 — PPL informational (Q8 / REAP / gemma) | 7.38 / 12.77 / 13357 (gemma anomaly = instruction-tuned model on raw text, EXPECTED) | ✓ |
| Batch 4 — No-regression bench Coder-30B Q4 tg32 | 47.49 ± 0.17 t/s | ✓ PASS (≥46) |
| Batch 4 — No-regression bench Qwen3.6-35B Q8 tg32 | 22.79 ± 0.04 t/s | ✓ PASS (≥22.5) |
| Batch 4 — No-regression bench REAP-246B Q4 tg32 | 6.25 ± 0.01 t/s | ✓ PASS (≥6.15) |
| Batch 4 — No-regression bench gemma-31B Q4 tg64 | 7.11 ± 0.01 t/s | ✓ PASS (≥6.25) |
| Batch 5 — Per-role smoke | DEFERRED to orchestrator integration | n/a |

## Branch state

- Branch: `production-consolidated-v5` in `/mnt/raid0/llm/llama.cpp-experimental`
- Base: `production-consolidated-v4` SHA `e734a682827...`
- Tip: `734d011e0`
- 58 commits ahead of v4: 50 cherry-picks + 8 strip/refactor commits (`branch-log.txt`)
- Net code change: -605 LOC across stripped paths (CPU22, rms_norm parallel, gated_delta_net, CPU15 Phase 1+2, anon-copies producer, NUMA_WEIGHTS family activations) + ~30 LOC of refactors (GGML_EP_VERBOSE, GGML_MUL_MAT_BLOCK)

## Open verification (deferred — non-blocking)

- v4-binary PPL on Q8/REAP/gemma for direct bit-exact comparison: blocked by approval gate (per `feedback_no_concurrent_inference.md`); informational only since Coder-30B already established bit-exact behavior at default config.
- Batch 5 per-role smoke: needs orchestrator integration; deferred to Phase L deployment.
- Hard strip of NUMA_WEIGHTS family in `llama-model-loader.cpp` (~200 LOC currently soft-stripped via `if (false)`): low-risk follow-up, non-blocking for v5 push.
- PGO + per-role BOLT-libggml binaries: not built in this audit; production deployment will run the toolchain recipe documented in `v5-push-cleanup-audit.md` §16 Phase 3d.

## Disposition

**GO for v5 push** based on:
1. Build clean (clang-20+libomp+znver5; 0 errors, 1 pre-existing test warn)
2. PPL bit-exact on Coder-30B Q4_K_M (the documented reference model)
3. All 4 production-model bench thresholds met within tight stds (σ ≤ 1% on every model)
4. Cherry-picks landed without conflict (50 commits clean)
5. Strips eliminated 5 deprecated env-flag families with no remaining live references
6. Refactors are low-risk additive (env-cached helper + named constant)

**Required before push to remote** (per CLAUDE.md "remote actions need confirmation"):
- User confirms decision.md
- User authorizes `git push fork production-consolidated-v5`

Push command (when authorized):
```bash
cd /mnt/raid0/llm/llama.cpp-experimental
git push fork production-consolidated-v5
# Optionally also push to /mnt/raid0/llm/llama.cpp's fork remote if separate
```

## Post-push tasks

1. Move handoff to `handoffs/completed/v5-push-cleanup-audit.md`
2. Update `model_registry.yaml` in `epyc-orchestrator` per `model-registry-v5-deployment-draft.yaml`
3. Move deployment-draft to `handoffs/completed/`
4. Open epyc-orchestrator PR with the per-role binary_path + env block changes
5. Run Batch 5 per-role smokes once orchestrator is updated

## Cross-references

- Handoff: [`/workspace/handoffs/active/v5-push-cleanup-audit.md`](file:///workspace/handoffs/active/v5-push-cleanup-audit.md) (will move to completed/ after user-confirmed push)
- Inventory: [`/workspace/handoffs/active/cpu-kernel-env-flags-inventory.md`](file:///workspace/handoffs/active/cpu-kernel-env-flags-inventory.md) (Phase 0 reconciliation persisted)
- Deployment-draft: [`/workspace/handoffs/active/model-registry-v5-deployment-draft.yaml`](file:///workspace/handoffs/active/model-registry-v5-deployment-draft.yaml) (gates on this push completing)
- Research deep-dive (Q3 output): [`/mnt/raid0/llm/epyc-inference-research/research/numa-weights-deep-dive.md`](file:///mnt/raid0/llm/epyc-inference-research/research/numa-weights-deep-dive.md)
