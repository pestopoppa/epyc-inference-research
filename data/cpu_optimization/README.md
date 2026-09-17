# CPU optimization campaign archive — 2026-04-23 → 2026-07-03

Provenance documentation added **2026-09-15** (NIB2-73a residual: this campaign directory carried
no `README.md` and no `SHA256SUMS`, so the evidence-durability gate warned on it and a reader could
not tell what was measured or which claim it backs). The measurement files themselves were committed
between 2026-04-24 and 2026-07-03 across **65 commits** and are unchanged; nothing was copied or
migrated by this commit.

**Read this first: `cpu_optimization` is not one campaign.** The name is a bucket. It holds **66
date-prefixed run bundles** from roughly thirty *distinct* investigations — the CPU1/CPU2 kernel
stack, CPU4 barrier coalescing, CPU11 PGO / CPU12 BOLT toolchain work, CPU15 expert parallelism,
CPU17 Sarathi-Serve, CPU21 libomp chunking, CPU22 work-stealing, CPU23 context regimes, CPU24
uncore attribution, CPU25 NUMA_MIRROR, the NPS4 topology work, MoE-Spec, MAB tree-shape selection,
slot promotion, the v5 kernel cleanup audit, three arch-class probes, and the 2026-05-28 decode
roofline audit. **Do not cite `data/cpu_optimization/` as a unit and do not attribute one
subdirectory's verdict to another.** The table below names each bundle, the commit that added it,
and the epyc-root document that owns it.

| | |
|---|---|
| scratch origin | **none established.** These bundles were committed from this path by per-session `data(cpu): …` commits; no run bundle here records a `/mnt/raid0/llm/tmp/` origin. `UNVERIFIED — whether any individual run was first written to scratch and copied in` (no migration record exists in the git history of this path). |
| measured | **2026-04-23 → 2026-07-03**, per the date prefix in each subdirectory name, corroborated by each bundle's adding-commit date (see the per-bundle table). Several bundles are *retroactive backfills* whose commit date postdates the measurement date they are named for — e.g. `2026-04-28-cpu2-q6k-full-ppl/` was added by `9675b223` dated **2026-04-27**, and `2026-04-24-nps4/` by `d19ff3d1` dated **2026-04-26**. Where the two disagree, the **directory-name date is the measurement date** and the commit date is when it was recorded. File mtimes are NOT used — in a fresh worktree they are checkout times (OBS-13). |
| carried (tracked) | **2,652 files, 17,880,101 bytes** (17.05 MiB) across 66 subdirectories; no tracked files at the top level |
| also on disk, gitignored | **23 files, 13,478,510,588 bytes (13.48 GB)** — `perf.data` profiles and BOLT `.so.bolted` / `.so.original` pairs under `2026-04-23/perf/`, `2026-04-26-cpu24/perfrecord/`, `2026-04-28-cpu12-bolt*/`, `2026-04-28-moe-spec-phase-1/bolt-v5/` and `2026-04-29-bolt-libggml-v5-redo/`. Reported, not restructured: the largest single file is `2026-04-28-cpu12-bolt-libomp/perf_reap.data` at 3.91 GB. These are correctly gitignored under the 2026-08-03 ruling that raw campaign substrate stays on local disk. |

## Registry claims this backs

`orchestration/model_registry.yaml` cites **only two** of the 66 bundles. Line numbers are as of
2026-09-15; the YAML key path is the stable reference.

- **L334** &nbsp;`kernel_audits[].bundle` → `2026-04-30-v5-cleanup-audit/` (15 files, 78,655 B).
  The bundle behind the `production-consolidated-v5` kernel audit row (`audit_date: 2026-04-30`,
  `tip_sha: 23bcd6aaf`, `base_sha: e734a6828`, handoff `handoffs/active/v5-push-cleanup-audit.md`,
  since moved to `completed/`). The registry's own summary: *"59 commits ahead of v4 (50 cherry-picks
  + 9 strip/refactor commits net ~-940 LOC of dead-by-default deprecated code)"*, stripping CPU22
  work-stealing, `GGML_RMS_NORM_PARALLEL`, `GGML_GDN_K_PER_HEAD`, CPU15 Phase 1+2 and the
  `GGML_NUMA_WEIGHTS` family. The bundle's own `2026-04-30-v5-cleanup-audit/README.md:6` records its
  status at capture time as **"validation gates running"** — i.e. the bundle is the audit *record*,
  and the pass verdict lives in the handoff, not in the bundle.
- **L1338** &nbsp;`server_mode.architect_general.throughput` (a comment inside that key's block) →
  `2026-05-04-qwen35-122b-arch-probe/` (32 files, 38,335 B). Named as the bundle behind the
  Qwen3.5-122B-A10B figures the registry relocated to `server_mode.architect_critic`, including the
  `12.19 t/s` 2026-05-04 NO-MTP probe. The bundle's `findings.md:4-5` states its own method —
  *"Probe B methodology, **n=5 reps**, canonical recipe (`numactl --interleave=all` + `taskset 0-95`
  + OMP env stack + `--mmap 0` + `-fa 1`)"* — and a tripwire (Coder-30B Q4_K_M `tg32 r=5`
  `47.86 ± 0.36 t/s`, canonical band 47–49). Phase-1 result, with units and σ: default v5
  `12.041 ± 0.037 t/s`, CPU1 stack `12.065 ± 0.024` (+0.21%), mbind off `12.195 ± 0.051` (+1.28%).

**The other 64 bundles are not registry evidence.** They are cited from epyc-root handoffs, `docs/`
and `wiki/`, which is why the durability gate reports them as neither resolved nor missing — they
are simply not in its citation set. The owner column below is the citation of record for each.

## What this proved

There is no single verdict for this directory, and writing one would be the flattening error the
table exists to prevent. Each bundle's own verdict is carried in its adding commit's subject line,
which for this campaign is unusually load-bearing: the commits say `— NOT DEPLOYABLE`,
`— DEFINITIVE NO-GO across both targets`, `— gate FAILED — closes track via test`,
`— both NEUTRAL/INCONCLUSIVE`, `— NO-GO via test`, `— WIN — verification-batch gate MET`,
`+8.9% does NOT generalize`, and so on. Those subjects are reproduced verbatim in the table. **A
majority of these bundles record a negative or inconclusive result**, and that is what they are for.

The most-cited single finding in this directory is **not** registry evidence:
`2026-05-28-decode-roofline/findings.md` (1 file, 10,841 B) is the CPU decode FLOPS/BW roofline
audit — the origin of the standing *compute-idle* conclusion. Its recorded verdict is in epyc-root
`handoffs/completed/fable5-findings-appendix-evidence-reports.md:890`: *"EVIDENCE:
…/2026-05-28-decode-roofline/findings.md:33-36 (FLOPS bounds), :54-57 (per-token cross-check),
:140-141 (addendum cross-check). **Strict >70%-of-peak BW gate NOT met** (healthy-projected 36.7%
theoretical / 48.9% practical) because gemma4 is 4B-active × Q4 small-traffic (:59-76)"*, and it is
the "Related" pointer at `handoffs/active/tidar-one-pass-variant-b.md:7`.

`UNVERIFIED — an owning handoff for 13 of the 66 bundles.` They have no reference anywhere in
epyc-root `handoffs/`, `docs/` or `wiki/` (searched against `origin/main`, 2026-09-15). They are
marked `**UNVERIFIED** — no epyc-root reference` in the table; for those, the adding commit subject
is the *only* attribution on record and is quoted as such. They are not orphans — each belongs to
the session named in its commit — but no document claims them.

`UNVERIFIED — per-bundle n and metric direction.` This README does not restate measurements from
individual bundles. Several carry their own `findings.md` / `SUMMARY.md` (19 of the 66 do) with n and
σ stated; the rest are raw `llama-bench` JSON plus logs. Quote a number only from the bundle's own
findings file or from the epyc-root document that owns it, with its unit and n.

## Per-bundle inventory

66 subdirectories. `files` and `bytes` count **tracked** files only. `added by` is the first commit
that added any file under that path (`git log --diff-filter=A`). `owned by` is the epyc-root document
that references the path, preferring an active handoff, then a completed/archived one, then
`docs/`, then `wiki/`.

| bundle | files | bytes | added by (commit subject = its verdict of record) | owned by |
|---|---:|---:|---|---|
| `2026-04-23/` | 40 | 160,840 | `dc4b3340` 2026-04-24 — CPU optimization benchmark data (Apr 23-24 session) | `nps-reboot-runbook` (completed) |
| `2026-04-24/` | 353 | 434,795 | `dc4b3340` 2026-04-24 — CPU optimization benchmark data (Apr 23-24 session) | `cpu-shape-specialized-gemv-decode` (active) (+7 more) |
| `2026-04-24-large-moe-baseline/` | 7 | 9,526 | `346660e8` 2026-04-24 — CPU15 Phase 0: large-MoE baseline measurements on NPS4 + auto-mbind + AVX-512BW + CPU1 stack | `large-moe-expert-parallelism-completed-through-2026-05-28` (completed) |
| `2026-04-24-nps4/` | 108 | 127,972 | `d19ff3d1` 2026-04-26 — data: 2026-04-24 NPS4 + 2026-04-26 L3aaN raw bench artifacts | `intra-process-tensor-parallel-decode-completed-through-2026-05-28` (completed) (+1 more) |
| `2026-04-24-q8-8x8-kernel/` | 5 | 7,292 | `d6759199` 2026-04-24 — Session 15 (2026-04-24): CPU2 AVX-512BW 8x8 Q8_0 kernel measurements | `cpu-shape-specialized-gemv-decode` (active) (+1 more) |
| `2026-04-24-q8-profile/` | 3 | 26,399 | `f9e8b3ca` 2026-04-24 — Session 15 part 4: perf profile of Qwen3.6-27B Q8_0 decode | `cpu-shape-specialized-gemv-decode` (active) (+1 more) |
| `2026-04-25-large-moe-ep-phase1/` | 5 | 35,115 | `6821b496` 2026-04-25 — CPU15 Phase 1 measurements: per-CCD expert sharding implemented but D3 gate fails | `large-moe-expert-parallelism-completed-through-2026-05-28` (completed) |
| `2026-04-25-large-moe-ep-phase2/` | 3 | 19,904 | `9dc4aacf` 2026-04-25 — CPU15 Phase 2: anonymous-mmap'd expert copies — implementation correct, throughput regresses 4.5% | `large-moe-expert-parallelism-completed-through-2026-05-28` (completed) |
| `2026-04-26-asymmetry/` | 40 | 31,816 | `721eb1a4` 2026-04-26 — data: 2026-04-26 asymmetry investigation raw bench artifacts | `model-registry-v5-deployment-draft` (active) (+2 more) |
| `2026-04-26-compounding/` | 20 | 17,755 | `42f1789c` 2026-04-26 — data: 2026-04-26 compounding-matrix raw bench artifacts | `frontier-f6-upstream-publication` (active) (+5 more) |
| `2026-04-26-cpu17/` | 6 | 6,526 | `aa789f05` 2026-04-26 — data: 2026-04-26 CPU17 Sarathi-Serve quick probe artifacts | `sarathi-serve-cpu-evaluation` (active) (+1 more) |
| `2026-04-26-cpu21/` | 38 | 40,796 | `f4497bf3` 2026-04-26 — data: 2026-04-26 CPU21 sweep + CPU24-deeper infra scripts | `cpu-benchmark-rigor-and-revalidation` (completed) |
| `2026-04-26-cpu24/` | 19 | 48,131 | `989239d0` 2026-04-26 — data: 2026-04-26 CPU24-narrow attribution perf-stat artifacts | `cpu-benchmark-rigor-and-revalidation` (completed) |
| `2026-04-26-l3aan/` | 40 | 21,238 | `d19ff3d1` 2026-04-26 — data: 2026-04-24 NPS4 + 2026-04-26 L3aaN raw bench artifacts | `nps-reboot-runbook` (completed) (+2 more) |
| `2026-04-26-nps4-restore/` | 10 | 4,181 | `4cc08d6d` 2026-04-26 — data: 2026-04-26 NPS4 post-revert smoke test artifacts | **UNVERIFIED** — no epyc-root reference |
| `2026-04-27-cpu2-prefetch/` | 13 | 53,520 | `83668207` 2026-04-27 — data: 2026-04-27 CPU2 Session 18 prefetch tuning artifacts | **UNVERIFIED** — no epyc-root reference |
| `2026-04-27-cpu2-q6k/` | 6 | 26,552 | `fbf311f6` 2026-04-27 — data: 2026-04-27 CPU2 Q6_K SIMD body smoke + 3-chunk PPL artifacts | **UNVERIFIED** — no epyc-root reference |
| `2026-04-27-cpu23/` | 20 | 49,689 | `91022d49` 2026-04-27 — data: 2026-04-27 CPU23 context-regime sweep artifacts | `cpu-benchmark-rigor-and-revalidation` (completed) |
| `2026-04-27-cpu25-numa-mirror/` | 9 | 20,290 | `d687db9e` 2026-04-27 — remediation phase 2.5: CPU20 retroactive artifact-bundle backfill for CPU21/23/24/25 | `cpu-benchmark-rigor-and-revalidation` (completed) |
| `2026-04-27-numa-mirror-phase0a/` | 6 | 38,580 | `54fc2f01` 2026-04-27 — data: 2026-04-27 NUMA_MIRROR Phase 0 validation artifacts | **UNVERIFIED** — no epyc-root reference |
| `2026-04-28-cpu-cross-architecture-sanity/` | 14 | 22,130 | `907bf4dc` 2026-04-27 — remediation phase 2.6: cross-architecture sanity coverage on dense/hybrid Qwen3.6-27B Q8_0 — closes peer-review finding #11 | `cpu-shape-specialized-gemv-decode` (active) (+2 more) |
| `2026-04-28-cpu11-pgo/` | 16 | 39,130 | `dd3d8b1c` 2026-04-28 — CPU11 PGO + CPU12 BOLT bundles — Phase 2.1 followup | `qwen36-benchmark-fixes` (completed) (+2 more) |
| `2026-04-28-cpu12-bolt/` | 25 | 871,436 | `dd3d8b1c` 2026-04-28 — CPU11 PGO + CPU12 BOLT bundles — Phase 2.1 followup | `cpu-inference-optimization-index-history-through-2026-06-19` (archived) (+1 more) |
| `2026-04-28-cpu12-bolt-libomp/` | 15 | 263,164 | `39e6adce` 2026-04-28 — CPU11 LTO + CPU12 BOLT-libomp extensions — both NEUTRAL/INCONCLUSIVE | `hardware-optimization` (wiki) |
| `2026-04-28-cpu2-q6k-full-ppl/` | 11 | 66,468 | `9675b223` 2026-04-27 — remediation phase 2.4: Q6_K AVX-512BW SIMD full 32-chunk WikiText-2 PPL gate — PASSED bit-exact | `cpu-shape-specialized-gemv-decode` (active) (+1 more) |
| `2026-04-28-cpu21-libomp-chunks/` | 32 | 34,729 | `52cf10c1` 2026-04-27 — remediation phase 2.1 (partial): CPU21 chunks 8/16 sweep + cross-model verification | `model-registry-v5-deployment-draft` (active) (+2 more) |
| `2026-04-28-cpu22-work-stealing/` | 19 | 51,037 | `bbbd9f26` 2026-04-27 — remediation phase 3: CPU22 work-stealing prototype + gate FAILED — closes track via test | `cpu-dynamic-moe-load-balancing` (completed) (+2 more) |
| `2026-04-28-cpu23-interference-metrics/` | 99 | 1,648,798 | `259a168b` 2026-04-27 — remediation phase 2.2: CPU23 minimum-gate fill (4 regimes × 4 metrics × 3 proxies) — closes peer-review CRITICAL finding #1 | `cpu-context-regime-coverage` (completed) (+3 more) |
| `2026-04-28-cpu24-minimax-and-dense/` | 15 | 39,344 | `a5340b69` 2026-04-27 — remediation phase 2.3: CPU24 attribution + MiniMax + dense + 2-rep stability — closes peer-review HIGH finding #4 | `cpu-uncore-fabric-attribution` (completed) (+1 more) |
| `2026-04-28-moe-spec-phase-1/` | 119 | 850,273 | `34747f8a` 2026-04-28 — data(cpu): MoE-Spec Phase 1 prototype measurement bundle (WIN — verification-batch gate MET) | `moe-spec-cpu-spec-dec-integration` (active) (+2 more) |
| `2026-04-29-bolt-libggml-v5-redo/` | 17 | 895,487 | `c10249b7` 2026-04-28 — data(cpu): Phase 3 #1 BOLT-libggml v5 redo — NOT DEPLOYABLE | **UNVERIFIED** — no epyc-root reference |
| `2026-04-29-clean-host-verification/` | 1 | 3,105 | `8d1914eb` 2026-04-29 — data(cpu): multi-arch coverage rerun (aborted) + clean-host verification (FAILED — host throttle) | **UNVERIFIED** — no epyc-root reference |
| `2026-04-29-coder-moe-spec-10rep-alternated/` | 18 | 28,134 | `27057052` 2026-04-28 — data(cpu): Phase 3 #2 + #3 — Coder 10-rep alternated + Q8 frontdoor MoE-Spec | **UNVERIFIED** — no epyc-root reference |
| `2026-04-29-cpu4-op-coalesced-barriers-phase0/` | 2 | 16,667 | `0f01c8e6` 2026-04-29 — data(cpu): CPU4 op-coalesced barriers Phase 0 — manual op-chain analysis, GATE PASSED | `cpu4-deferred-avenues-design-note` (completed) |
| `2026-04-29-cpu4-op-coalesced-barriers-phase1/` | 29 | 115,594 | `06e8dade` 2026-04-29 — data(cpu): CPU4 Phase 1 op-coalesced barriers — NO-GO via test | `cpu4-deferred-avenues-design-note` (completed) |
| `2026-04-29-hybrid-ssm-slot-promotion-phase-0/` | 7 | 8,658 | `ce9bc1a4` 2026-04-28 — data(cpu): MAB tree-selector Phase 0 NO-GO + slot-promotion Phase 0 GO bundles | `hybrid-ssm-slot-promotion-spec-dec` (completed) |
| `2026-04-29-mab-phase-0-prime-prime-replication/` | 368 | 3,138,430 | `1f1e4615` 2026-04-29 — data(cpu): MAB Phase 0'' n=90 high-rep replication — DEFINITIVE NO-GO across both targets | `mab-tree-shape-selector` (completed) (+1 more) |
| `2026-04-29-mab-tree-selector-phase-0/` | 25 | 192,090 | `ce9bc1a4` 2026-04-28 — data(cpu): MAB tree-selector Phase 0 NO-GO + slot-promotion Phase 0 GO bundles | `speculative-decoding` (wiki) |
| `2026-04-29-moe-dynamic-expert-phase-0/` | 7 | 11,928 | `f7ad0c7c` 2026-04-28 — data(cpu): Phase 3 #4 — moe-dynamic-expert-selection Phase 0 entropy probe NEGATIVE | `moe-dynamic-expert-selection` (completed) |
| `2026-04-29-moe-spec-q8-dense/` | 16 | 27,509 | `27057052` 2026-04-28 — data(cpu): Phase 3 #2 + #3 — Coder 10-rep alternated + Q8 frontdoor MoE-Spec | **UNVERIFIED** — no epyc-root reference |
| `2026-04-29-multi-arch-coverage/` | 21 | 24,531 | `9b2f8c47` 2026-04-29 — data(cpu): multi-arch coverage probe — CPU1 + CPU2 mbind on dense Q8 / hybrid SSM / dense Q4 (first-pass) | `model-registry-v5-deployment-draft` (active) |
| `2026-04-29-multi-arch-coverage-canonical/` | 15 | 18,694 | `f0b7d7b4` 2026-04-29 — data(cpu): multi-arch coverage matrix + Probe B workload-shape under canonical OMP recipe | `model-registry-v5-deployment-draft` (active) |
| `2026-04-29-multi-arch-coverage-rerun/` | 14 | 13,125 | `8d1914eb` 2026-04-29 — data(cpu): multi-arch coverage rerun (aborted) + clean-host verification (FAILED — host throttle) | **UNVERIFIED** — no epyc-root reference |
| `2026-04-29-numa-parallel-ceiling/` | 13 | 11,882 | `11a4da72` 2026-04-28 — data(cpu): NUMA-parallel ceiling probe — Phase 2 gate MET (6.10×) | `hybrid-ssm-slot-promotion-spec-dec` (completed) |
| `2026-04-29-numa-quarter-pin-phase-1-1a/` | 49 | 362,182 | `a52abf96` 2026-04-28 — data(cpu): Phase 1.1 NUMA-parallel verify — FOUNDATION ONLY, two blockers identified | `hybrid-ssm-slot-promotion-spec-dec` (completed) |
| `2026-04-29-post-reboot-tripwire/` | 29 | 20,634 | `3405709a` 2026-04-29 — data(cpu): post-reboot reproducibility tripwire — RESOLVED via OMP env stack | `cpu-kernel-env-flags-inventory` (completed) |
| `2026-04-29-remediation-phase-A-cpu4/` | 5 | 7,198 | `92dfd653` 2026-04-29 — data(cpu): Remediation Phases A-D — re-validate prior closures under canonical recipe | `cpu4-deferred-avenues-design-note` (completed) (+1 more) |
| `2026-04-29-remediation-phase-B-slot-promotion/` | 18 | 133,264 | `92dfd653` 2026-04-29 — data(cpu): Remediation Phases A-D — re-validate prior closures under canonical recipe | **UNVERIFIED** — no epyc-root reference |
| `2026-04-29-remediation-phase-C-mab/` | 551 | 4,816,862 | `92dfd653` 2026-04-29 — data(cpu): Remediation Phases A-D — re-validate prior closures under canonical recipe | `mab-tree-shape-selector` (completed) (+1 more) |
| `2026-04-29-remediation-phase-D-cpu22/` | 9 | 9,784 | `92dfd653` 2026-04-29 — data(cpu): Remediation Phases A-D — re-validate prior closures under canonical recipe | **UNVERIFIED** — no epyc-root reference |
| `2026-04-29-slot-promotion-phase-1/` | 16 | 107,253 | `977813ca` 2026-04-28 — data(cpu): Slot-promotion Phase 1.0 GATE MET — heap-spec works on hybrid Delta Net | `hybrid-ssm-slot-promotion-spec-dec` (completed) |
| `2026-04-29-workload-shape-canonical/` | 30 | 29,486 | `f0b7d7b4` 2026-04-29 — data(cpu): multi-arch coverage matrix + Probe B workload-shape under canonical OMP recipe | `model-registry-v5-deployment-draft` (active) |
| `2026-04-29-workload-shape-coverage/` | 1 | 4,267 | `ee74d79b` 2026-05-06 — data(may4-6): benchmark runs, cpu-opt logs, preflight captures | **UNVERIFIED** — no epyc-root reference |
| `2026-04-30-divergent-tree-sweep/` | 25 | 854,909 | `d1b66573` 2026-04-28 — data(cpu): Phase 1.1 dispatcher v1 — divergent-tree sensitivity sweep | `hybrid-ssm-slot-promotion-spec-dec` (completed) (+2 more) |
| `2026-04-30-hybrid-ssm-next80b-followup/` | 12 | 14,274 | `c07e37e8` 2026-04-30 — data(cpu): Hybrid SSM follow-up on Qwen3-Next-80B-A3B — +8.9% does NOT generalize | `model-registry-v5-deployment-draft` (active) |
| `2026-04-30-mab-phase-0-prime-sampling/` | 66 | 433,196 | `398f9de5` 2026-04-28 — data(cpu): two probes — slot-promotion dense-target (NO-GO) + MAB Phase 0' sampling (INCONCLUSIVE signal) | `mab-tree-shape-selector` (completed) (+1 more) |
| `2026-04-30-slot-promotion-dense-target/` | 18 | 879,261 | `398f9de5` 2026-04-28 — data(cpu): two probes — slot-promotion dense-target (NO-GO) + MAB Phase 0' sampling (INCONCLUSIVE signal) | **UNVERIFIED** — no epyc-root reference |
| `2026-04-30-state-sync-cost-probe/` | 33 | 278,149 | `c18560e3` 2026-04-28 — data(cpu): Phase 1.1 dispatcher v1 — state-sync probe + canonical 3x2 measurement | `hybrid-ssm-slot-promotion-spec-dec` (completed) (+2 more) |
| `2026-04-30-v5-cleanup-audit/` | 15 | 78,655 | `3636e48b` 2026-04-30 — data+research+orchestration: v5 cleanup audit bundle + deep-dive + registry annotation | `v5-push-cleanup-audit` (completed) (+2 more) |
| `2026-05-04-q6k-default-on-validation/` | 44 | 168,506 | `8b59a1fb` 2026-05-04 — data(cpu_optimization 2026-05-04): Q6_K validation + Qwen3.5-122B-A10B Probe B bundles | `cpu-shape-specialized-gemv-decode` (active) (+4 more) |
| `2026-05-04-qwen35-122b-arch-probe/` | 32 | 38,335 | `8b59a1fb` 2026-05-04 — data(cpu_optimization 2026-05-04): Q6_K validation + Qwen3.5-122B-A10B Probe B bundles | `model-registry-v5-deployment-draft` (active) (+5 more) |
| `2026-05-04-reap246b-arch-probe/` | 13 | 15,609 | `bd5dbf83` 2026-05-04 — data(2026-05-04): REAP-246B-A35B Q4_K_M Probe B — default v5 confirmed | `model-registry-v5-deployment-draft` (active) (+2 more) |
| `2026-05-28-decode-roofline/` | 1 | 10,841 | `8cc1e879` 2026-05-28 — data: 2026-05-28 decode-roofline bundle (gemma4-26B-A4B Q4_K_M) | `tidar-one-pass-variant-b` (active) (+1 more) |
| `2026-05-30-v4-throughput-gate-provisional/` | 3 | 22,969 | `d29ed5b4` 2026-05-30 — v4_throughput_gate: fix sed regex + commit provisional FAIL result | `deepseek-v4-flash-cpu-port` (completed) |
| `2026-07-03-amd-perf-counter-preflight/` | 2 | 2,222 | `ad9b73aa` 2026-07-03 — Add AMD perf counter preflight | `cpu-prefill-compute-large-models` (active) (+2 more) |
| `pre-nps4-freeze/` | 11 | 18,985 | `dc4b3340` 2026-04-24 — CPU optimization benchmark data (Apr 23-24 session) | `nps-reboot-runbook` (completed) (+1 more) |

## Notes on two naming traps

- `2026-04-24/` (353 files) and `2026-04-23/` (40 files) are the two **undifferentiated** session
  dumps from before the per-track naming convention started. Between them they are referenced by
  seven different epyc-root documents, so a path under `2026-04-24/` cannot be attributed to a
  single track from the directory name alone. Resolve it through the referencing document.
- `2026-04-29-multi-arch-coverage/`, `…-multi-arch-coverage-rerun/` and
  `…-multi-arch-coverage-canonical/` are **three different runs**, one of which (`-rerun`) was
  **aborted** — its adding commit says so verbatim. Likewise `2026-04-29-mab-tree-selector-phase-0/`,
  `2026-04-30-mab-phase-0-prime-sampling/` and `2026-04-29-mab-phase-0-prime-prime-replication/` are
  three escalating replication rounds ending in a `DEFINITIVE NO-GO`; the earlier two are not
  independent support for anything.

## Content notes

No credential-shaped strings were found in the tracked files. One email address is present:
the **operator's own**, in `2026-04-30-v5-cleanup-audit/README.md:27`, attributing his own decisions
in that audit's phase table. It is first-party, not third-party PII, so the `WITHHELD` precedent in
`data/paddleocr_vl_receipt_extract_20260717T194415Z/` does **not** apply and nothing here is
withheld. The tracked files otherwise carry first-party operational detail — the hostname, local
absolute paths under `/mnt/raid0/llm/`, local commit SHAs, `perf` symbol names from local builds,
and OS PIDs in process dumps.

## Integrity

`SHA256SUMS` seals all 2,652 tracked files. This directory's own `README.md` and `SHA256SUMS` are
deliberately not in it (documentation is not evidence, and hashing the README would make every doc
edit break the seal), following the 2026-08-02 precedent in
`data/paddleocr_vl_receipt_extract_20260717T194415Z/`. The **nested** bundle readme at
`2026-04-30-v5-cleanup-audit/README.md` **is** sealed — it predates this work, is registry-cited
evidence in its own right, and is not documentation added by this commit. Verify from the repository
root:

```bash
cd /mnt/raid0/llm/epyc-inference-research && sha256sum -c data/cpu_optimization/SHA256SUMS
```
