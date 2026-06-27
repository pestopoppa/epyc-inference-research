# epyc-inference-research Handoff Index

**Purpose**: machine-readable coordination point for benchmark, dataset, and research-artifact work owned by this repository.

This index tracks research-repo tasks only. Production routing, AutoPilot authority, and live stack flips remain governed from `epyc-root` and `epyc-orchestrator`.

## Prioritized Task List

- [ ] **P0 - Preserve active evidence windows before launching clean-window jobs.**
  Confirm no AutoPilot, X-MAS A/B, DS-E1, or manual bench run is active before starting any new live inference workload. Use the root/orchestrator handoff state as authority.

- [ ] **P1 - DS-E1 KV production measurements.**
  Run the decision-grade 2K/8K/32K KV-size measurements only in a clean inference window, then write the result packet under `data/` and update the DS-E1 root handoff.

- [ ] **P1 - X-MAS repaired-policy A/B.**
  Run the held-out `incumbent_constrained_cheapfirst_v2` policy A/B only after the host is quiet. Preserve row-level `xmas_meta` and summarize promotion status from evidence, not role names.

- [ ] **P1 - N5 frontdoor drafter alpha retest.**
  Retest the aligned qwen35/frontdoor drafter after `check_draft_compatibility.py` passes for the selected drafter. Do not reuse qwen2-tokenizer acceptance bins.

- [ ] **P2 - N14 prompt/sampling quality certification.**
  Co-schedule the canonical greedy-to-sampled bench with the next kernel-era clean window, then report whether the seeded model-card sampling defaults preserve quality.

- [ ] **P2 - A9 priority collection batches.**
  Generate or collect the remaining high-priority source-family and suite records without overlapping active evidence windows.

- [ ] **P2 - Research repository readiness L4 follow-up.**
  Add lightweight generated-doc, health/session, analysis-report, and security-audit surfaces when they match real workflows. Avoid placeholder automation.

- [ ] **P3 - Retire or refresh stale January orchestration task notes.**
  `research/NEXT_ORCHESTRATION_TASKS.md` is historical and predates the split-repo production stack. Either archive it or replace it with links to current root/orchestrator handoffs.

## Dependency Graph

```text
P0 clean-window guard
  -> P1 DS-E1 KV measurements
  -> P1 X-MAS repaired-policy A/B
  -> P1 N5 frontdoor drafter alpha retest
  -> P2 N14 prompt/sampling quality certification
  -> P2 A9 priority collection batches

P1/P2 evidence artifacts
  -> root/orchestrator handoff updates
  -> production gate decisions outside this repository

P2 readiness L4 follow-up
  -> stronger local validation/reporting
  -> safer future research automation
```

## Cross-Cutting Concerns

- **Measurement isolation**: live inference tasks can poison each other. Prefer one clean-window workload at a time unless the parent handoff explicitly calls for concurrent load.
- **Role drift**: benchmark results must be keyed by model, quant, flags, kernel era, and host regime; roles are mutable aliases.
- **Kernel era**: v6+iqk changed CPU-kernel performance characteristics. New results need explicit era/provenance fields or clear notes that they are post-cutover only.
- **Tokenizer compatibility**: speculative decoding acceptance data is invalid when drafter and target tokenizers are incompatible.
- **Repository boundary**: this repo produces evidence and tooling. Production flips happen in `epyc-orchestrator` after root handoff acceptance.

## Reporting Instructions

After completing any task:

1. Store raw outputs under an appropriate `data/` subdirectory.
2. Add or refresh the smallest summary document needed for future replay.
3. Record the exact command, commit, model path, quant, flags, sample size, and host-cleanliness condition.
4. Update the corresponding `epyc-root` handoff/progress entry.
5. Commit research artifacts by pathspec, leaving unrelated runtime outputs untouched.

## Key File Locations

- `scripts/benchmark/ds_e1_kv_measurements.sh` - DS-E1 measurement harness.
- `scripts/benchmark/n5_frontdoor_drafter_retest.sh` - aligned drafter retest harness.
- `scripts/research/xmas_function_axis_sweep.py` - X-MAS function-axis sweep runner.
- `scripts/research/xmas_winner_table.py` - X-MAS winner-table compiler.
- `scripts/benchmark/bench_canonical.sh` - canonical benchmark entry point.
- `scripts/benchmark/clean_window_manifest.py` - clean-window manifest helper.
- `orchestration/model_registry.yaml` - full research registry.
- `docs/reference/benchmarks/RESULTS.md` - benchmark result summary.
