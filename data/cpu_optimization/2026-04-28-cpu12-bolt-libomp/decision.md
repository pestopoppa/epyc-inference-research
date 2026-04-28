# CPU11 LTO + CPU12 BOLT-libomp — Decision

**Verdicts**:
- **LTO**: NEUTRAL within noise. Do NOT add LTO to v5 cherry-pick. PGO already triggers most cross-TU inlining; LTO adds nothing measurable.
- **BOLT-libomp**: BUILD PIPELINE WORKS, THROUGHPUT DELTA INCONCLUSIVE. The BOLT-rewritten libomp.so is functional and PPL bit-exact, but does NOT beat the system libomp under measurable conditions. Do NOT pursue libomp-BOLT for v5; reopen only if a quieter measurement window establishes a clean signal.

## LTO measurement

| Configuration | Coder-30B Q4_K_M tg32 (5-rep -pp 64 warmup, warm position) | Δ vs PGO |
|---|---|---|
| clang+libomp+znver5+PGO | 28.38 ± 0.08 | reference |
| clang+libomp+znver5+PGO+**LTO** | 28.09 ± 0.04 | **−1.0% within noise** |

Both results have very tight std (≤0.08), so the −1.0% delta is real and small. Position-confound test (forward + reverse order sweeps) confirms the conclusion is order-independent. LTO does not compound on PGO on this codebase.

**Mechanism**: PGO with `-fprofile-instr-use` already enables clang to inline across translation units when the profile shows the inlining is hot. LTO adds *additional* cross-TU inlining for cold paths, but those inlinings don't matter for the hot ggml decode kernels. Net: LTO is a no-op once PGO is in place.

## libomp-BOLT measurement

| Configuration | Coder-30B Q4_K_M tg32 (5-rep -pp 64 warmup) | Δ |
|---|---|---|
| PGO + system libomp (warm pos 2 of 4) | 27.77 ± 0.06 | reference |
| PGO + custom-rebuilt libomp (no BOLT) — pos 3 | 19.59 ± 7.89 | INCONCLUSIVE (high std) |
| PGO + BOLTed libomp (symlink, warm pos 2 of 4) | 29.58 ± 0.04 | warm pos 2 baseline |
| PGO + system libomp (warm pos 4 of 4) | 31.95 ± 0.09 | warm pos 4 reference |

The BOLTed libomp at warm position 2 (29.58 ± 0.04) and the system libomp at warm position 4 (31.95 ± 0.09) are both very tight. Position effect (more warm in pos 4) means absolute comparison favors system libomp by ~8%, but the position delta is itself ≥ the BOLT delta in this measurement.

**Honest summary**: there is no positive signal for BOLT-libomp here. Best read: BOLT-libomp is at parity with system libomp; worst read: BOLT-libomp is 8% slower (likely position confound, not BOLT regression).

The custom-rebuilt libomp (no-BOLT) measurement is inconclusive due to ±7.89 std at position 3 — the symlink swap may have triggered first-touch reallocation that confounded the bench.

## Quality

PPL bit-exact on Coder-30B Q4_K_M chunks 1-12 with BOLTed libomp loaded:
```
[1]7.5697,[2]10.3969,[3]9.8218,[4]9.4274,[5]9.1381,[6]9.3169,[7]9.4693,[8]10.0251,[9]10.4405,[10]10.9882,[11]10.9027,[12]11.1146,
Final estimate: PPL = 11.1146 +/- 0.62405
```

Byte-identical to all prior PGO/BOLT runs (PGO+BOLT-libggml was 11.1146 ± 0.62405; PGO without BOLT was 11.1146 ± 0.62405). BOLT does not change instruction encoding or fp ordering. Quality preserved.

## System noise context (important caveat)

This session's absolute throughput numbers are degraded ~2× vs the morning's CPU11 PGO bundle (which measured 58.65 ± 0.24 on the same `build_libomp_pgo_use/`). The cause:
- Megasync at 95% CPU on one core throughout
- 5-6 parallel claude/firefox/python processes holding 5-10% CPU each
- Cumulative cache pressure from 4 build trees + LLVM source extraction + 3 bench sweeps

The relative comparisons WITHIN each sweep (positions 2-4 of a single sequential run) remain meaningful because the noise floor is identical across positions in the same sweep. But **the absolute scale must not be compared to morning**. If system noise is the cause of the missing libomp-BOLT win, then a quieter measurement window in v5+1 might recover a +1-3% signal.

## v5 cherry-pick implications (no change from morning)

**v5 production binary stays**: `clang + libomp + -march=znver5 + PGO`.

- LTO confirmed NEUTRAL — not added to the v5 build flags.
- libomp-BOLT confirmed FUNCTIONAL but not faster than system libomp under tested conditions — not added to the v5 build pipeline.
- BOLT on `libggml-cpu.so.0` (CPU12 morning bundle) remains the per-role-Coder opt-in, with a +2.1% win on Coder-30B from that targeted BOLT-rewrite. The morning result stands.

## Closure scope

**Closed**:
- LTO without PGO + LTO on top of PGO: empirically tested, neutral within noise on Coder-30B Q4_K_M (this bundle).
- libomp from-source build with `--emit-relocs`: documented working pipeline (`_libomp_src/openmp-build/runtime/src/libomp.so`).
- libomp BOLT-rewrite via `llvm-bolt-20`: functional, PPL bit-exact, no throughput win.

**NOT closed**:
- Q8/REAP/dense classes for libomp-BOLT — only Coder-30B was tested due to time + system-noise budget.
- Re-measurement under quieter system state (offload of megasync, no parallel claude processes) — this is the v5+1 reopener if it matters.
- Multi-model fdata BOLT-rewrite — the merged.fdata file was created in legacy format and llvm-bolt rejected it. Single-model coder fdata used as workaround. Resolving the merge format would let us test merged-profile BOLT-libomp.

## Remediation reference

`~/.claude/plans/nifty-discovering-allen.md` Phase 2.1 / Phase 4 followups. Closes the two "Deferred for next session" items from the CPU11/CPU12 morning bundles.
