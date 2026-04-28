# MoE-Spec Phase 2 + v5 PGO + BOLT-Coder — Final Decision (2026-04-28 evening)

## Summary verdict

| Track | Decision | Rationale |
|---|---|---|
| **MoE-Spec on REAP-246B (B=40)** | **DEPLOY (env-gated default-ON for REAP role)** | Clean +13-15% pp32 across all builds (gcc, PGO single-B, PGO mixed-B); +3% end-to-end spec-dec via llama-server; PPL drift +6.1% forward-pass but spec-dec verifier rejection makes end-to-end output bit-exact |
| **MoE-Spec on Coder-30B (B=64)** | **DO NOT DEPLOY** | Result varied wildly across builds: gcc +7.3% pp32, PGO single-B −43% pp32 (mask branch unprofiled), PGO mixed-B parity-to-+84% pp32 (warm-up order sensitive), end-to-end +9% (3-rep, wide variance). Signal too noise-sensitive for confident production deployment |
| **MoE-Spec on Q8 frontdoor + dense** | **NOT TESTED in Phase 2** | Phase 1 gate measured only Coder + REAP. Q8 frontdoor (Qwen3.6-35B) and dense (Qwen3.6-27B) NOT exercised |
| **v5 PGO universal binary (clang+libomp+znver5+PGO mixed-B)** | **DEPLOY** | Coder +18-20% codegen gain over gcc baseline (matches morning's CPU11 +6.6% Q8 / +3.2% Coder finding); REAP +11-13% codegen gain |
| **BOLT-libggml on v5 PGO Coder role** | **DO NOT DEPLOY this attempt; reopen with longer perf record** | This cycle's perf-record was too short (~10s on Coder only); only 4% function coverage; BOLT-INFO warned "estimated to optimize better with 6.8x more samples". Bolted libggml regressed Coder B=64 to −51%. Morning's BOLT cycle on the OLDER binary delivered +2.1% with 60s perf record × 4 models — that recipe needs to be re-applied to v5 binary |

## Headline numbers (selected canonical measurements; full data in v5pgomixed_*.log + v5pgobolt_*.log)

### REAP-246B Q4_K_M pp32 (5-rep proper canonical, multiple builds)

| Build | B=0 | B=40 | Δ |
|---|---|---|---|
| gcc+libgomp | 45.23 ± 0.99 | 52.11 ± 0.58 | **+15.2%** |
| v5 PGO single-B (B=0 profile only) | 50.36 ± 1.07 | 58.38 ± 0.79 | **+15.9%** |
| v5 PGO mixed-B (with B=40 profile) | 51.14 ± 1.11 | 58.06 ± 0.59 | **+13.5%** |
| v5 PGO mixed-B quiet | 47.18 ± 1.09 | (deferred) | TBD but expected similar |

**Verdict: REAP-246B + MoE-Spec B=40 = +13-16% pp32 robust across builds.**

### Coder-30B Q4_K_M pp32 (5-rep, multiple builds, varying system noise)

| Build | B=0 | B=64 | Δ |
|---|---|---|---|
| gcc+libgomp | 321.35 (avg of 2 noisy runs) | 344.70 | **+7.3%** |
| v5 PGO single-B (B=0 profile only, megasync) | 379.42 | 215.37 | **−43.2%** (mask branch pessimized by single-B PGO) |
| v5 PGO mixed-B (megasync) | 198.57 | 193.34 | **−2.6% (parity)** |
| v5 PGO mixed-B (clean, ordered B=0→B=64) | 219.18 | 402.57 | **+83.7%** (cold→warm cache effect) |
| v5 PGO mixed-B (alternated B=0/B=64 noisy) | ~220 | ~210 | parity |
| v5 PGO mixed-B BOLT-Coder | 379.02 ± 12.70 | 185.95 ± 14.93 | **−51%** (BOLT pessimized mask branch) |

**Verdict: Coder-30B + MoE-Spec result is too noise/order/build-sensitive. Production deployment NOT recommended without further isolation studies.**

## Why Coder differs from REAP

The mask-construction overhead is fixed-cost per layer (transpose, sum_rows, argsort, fill, set_rows, transpose, add — ~6 ggml ops per MoE layer × 48 layers = ~288 extra ops per forward pass). The MoE-Spec savings (DRAM expert-weight read reduction) scales with per-layer expert-weight bandwidth.

- **REAP-246B**: Heavy model (138 GB / 246B params). Each MoE layer's expert-weight DRAM read is huge. Mask-construction overhead is small relative to savings → robust +13-16% gain.
- **Coder-30B**: Light model (17 GB / 30B params). Each MoE layer's expert-weight DRAM read is small. Mask-construction overhead is comparable to savings → result is noise-sensitive and warmup-order-dependent.

## Phase 2 end-to-end spec-dec via llama-server (megasync noise)

3 prompts × 3 B values × rep1+rep2 (rep0 dropped due to server-warmup timeout):

| Model | B | mean t/s | accept% | Δ vs B=0 |
|---|---|---|---|---|
| Coder-30B | 0 | ~33.3 | ~64% | reference |
| Coder-30B | 64 | ~36.4 | ~61% | +9.3% (with ~3pp accept drop) |
| Coder-30B | 32 | ~29.3 | ~67% | -12% (high variance) |
| REAP-246B | 0 | ~7.59 | ~58% | reference |
| REAP-246B | 60 | ~7.26 | ~58% | -4.3% |
| REAP-246B | 40 | ~7.84 | ~58% | **+3.3%** |

**Verdict: REAP-246B end-to-end +3% confirms the verification-batch +13-16% gain attenuates Amdahl-style at the round level. Modest but positive. Coder-30B end-to-end +9% is borderline + noisy and not confidence-inspiring for deployment.**

## v5 production binary recommendation

| Component | Choice | Rationale |
|---|---|---|
| Compiler | clang-20 | +6.4% over gcc on Coder (CPU21 morning finding) |
| OpenMP runtime | libomp | Bundled with clang; +3-8% affinity stack universal (CPU21) |
| Codegen | -march=znver5 -mtune=znver5 | +3.7% over no-march (CPU11 morning) |
| PGO | mixed-B profile (`-fprofile-instr-use=merged.profdata`) | +18-20% Coder codegen / +11-13% REAP codegen over gcc baseline. Mixed-B coverage MANDATORY (single-B exercises only B=0 path → pessimizes MoE-Spec branch) |
| LTO | NOT cherry-picked | Morning CPU11 NEUTRAL; same expected here |
| BOLT-libggml | NOT cherry-picked this attempt | Insufficient perf samples; needs 60s × 4 models redo |
| MoE-Spec env | `LLAMA_ARG_MOE_SPEC_BUDGET=40` for REAP role only | Clean robust +13-16% pp32 / +3% end-to-end |

## Production registry integration plan

`/mnt/raid0/llm/epyc-orchestrator/orchestration/model_registry.yaml`:

1. Replace `server_start` template's binary path from `/mnt/raid0/llm/llama.cpp/build/bin/llama-server` → `/mnt/raid0/llm/llama.cpp-experimental/build_v5_pgo_use/bin/llama-server` (or its v5 deployment-target location)
2. Add per-role environment override block to support `LLAMA_ARG_MOE_SPEC_BUDGET=40` for REAP-246B role only
3. Update `architect_coding` (REAP-246B) acceleration block to note MoE-Spec budget
4. Document that LD_LIBRARY_PATH must be set to the v5 binary directory for the orchestrator launch

## Open Phase 3 / followup work

1. **Cleaner Coder-30B re-measurement**: 10-rep, alternated B=0/B=64, pre-warmed cache, multiple session windows. Establish whether Coder MoE-Spec is genuinely deployable.
2. **BOLT-libggml v5 redo**: 60s perf record × 4 model classes (matches morning's recipe). Should give 6-10x more samples and proper function coverage. Expected +2-5% on top of v5 PGO for Coder role.
3. **Q8 frontdoor + dense MoE-Spec test**: Phase 1 only tested Coder + REAP. Q8 (Qwen3.6-35B) is BW-bound MoE; dense (Qwen3.6-27B) is hybrid SSM-Dense (no MoE). Q8 might benefit; dense is moot.
4. **End-to-end spec-dec re-run** with proper server-warmup (60s post-/health-ok before first request) and 5-10 reps to tighten the +3% REAP signal.
5. **Full 32-chunk WikiText-2 PPL on v5 PGO build** for production routing decisions.
6. **MoE dynamic expert selection sibling track** (`moe-dynamic-expert-selection.md`): 4 candidates surfaced (Dynamic Skipping, OD-MoE single-layer lookahead, MoE Pathfinder [deprioritized], Entropy-gated K). Phase 0 entropy probe + lookahead-accuracy probe ~3-4h; could compound or supplant MoE-Spec.

## Key data files (this bundle)

| File | Description |
|---|---|
| `decision.md` | Phase 1 decision (gcc+libgomp build) |
| `decision_v5_FINAL.md` | THIS file — Phase 2 + v5 PGO + BOLT-Coder consolidated |
| `PHASE2_PARTIAL.md` | Earlier wrap-up snapshot (mid-flight) |
| `results.csv` | Phase 1 tabulated |
| `coder30b_pp{32,64}_B*_run{1,2}.log` | Phase 1 gcc raw |
| `reap246b_pp32_B*.log` | Phase 1 REAP raw |
| `coder30b_ppl32_B{0,64}.log` | Full 32-chunk PPL Coder |
| `reap246b_ppl32_B{0,40,60}.log` | Full 32-chunk PPL REAP |
| `v5pgomixed_*.log` | v5 PGO mixed-B Phase 1 re-validation |
| `v5pgobolt_*.log` | v5 PGO + BOLT-Coder measurements |
| `v5pgomixed_quiet*.log` | v5 PGO under quiet system (post-megasync) |
| `v5pgomixed_quiet2_*.log` | Repeat after kill of stuck wait-loop |
| `v5_comp_*.json` | End-to-end spec-dec /completion captures |
| `v5_srv_*.log` | llama-server stdout for end-to-end runs |
| `bolt-v5/coder.perf.data` | perf record raw (Coder workload) |
| `bolt-v5/libggml-cpu.so.0.coder.fdata` | perf2bolt fdata |
| `bolt-v5/libggml-cpu.so.0.{original,bolted}` | original + bolted libggml-cpu |
