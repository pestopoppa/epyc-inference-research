# v5 Kernel Push — Cleanup Audit Bundle

**Date**: 2026-04-30
**Author**: pestopoppa (executed under Claude Opus 4.7 1M-context auto-mode session)
**Branch**: `production-consolidated-v5` in `/mnt/raid0/llm/llama.cpp-experimental`
**Status**: validation gates running

## Audit goal

Produce a clean, minimal, refactored `production-consolidated-v5` branch where every line is justified. Replace the noisier `feature/cpu-ep-inter-process` accumulated work (CPU1 + CPU2 + CPU15 + slot-promotion + CPU4 + falsified mechanisms + dead code) with a deliberate strip + cherry-pick + refactor pipeline.

User trigger (2026-04-30 wrap-up): *"shall we review what we plan to push into production to make sure code is clean, minimal and refactored? I don't want a sloppy mess."*

## Source handoff

Authoritative document: [`/workspace/handoffs/active/v5-push-cleanup-audit.md`](file:///workspace/handoffs/active/v5-push-cleanup-audit.md)

Key sub-documents:
- `cpu-kernel-env-flags-inventory.md` — env-flag inventory + reconciled cherry-pick scope (Phase 0 output)
- `model-registry-v5-deployment-draft.yaml` — staging file for per-role binary_path + env (gates on Phase 4 passing)

## Phases executed (this session)

| Phase | Status | Notes |
|---|---|---|
| 0 — Reconciliation | ✓ done | 35 KEEP / 11 STRIP-and-cherry-pick-then-refactor / 4 op-fusion canceling / 5 NUMA_MIRROR skipped = 55 commits in `v4..feature/cpu-ep-inter-process`. Both repos' v4 branches verified at SHA `e734a682827...`. |
| 1 — Decisions | ✓ done | All 5 reviewer questions resolved 2026-04-30 by daniele.pinna@gmail.com. Decision table filled. |
| 2a — TODO investigation | ✓ done | repack.cpp:3854 "TODO: this branch seems wrong" is in upstream IQ4_NL code (not Q8_0) and the branch is already commented out. NOT a v5 blocker. |
| 2b — Refactors | ✓ done | GGML_EP_VERBOSE gating (5 INFO fprintf wrapped) + GGML_MUL_MAT_BLOCK named constant (6 bare 16's). |
| 3a — Branch | ✓ done | `production-consolidated-v5` from `production-consolidated-v4`. |
| 3b — Cherry-picks | ✓ done | 50 commits cherry-picked clean, ZERO conflicts. |
| 3c — Strips | ✓ done | 8 atomic forward-style commits. -605 LOC net (CPU22, rms_norm, gdn, CPU15 P1+2, anon-copies, NUMA_WEIGHTS family). |
| 3d — Toolchain | ✓ reframed | NO code commit. Operator-applied at build time via CMake flags. |
| 4 — Validation | running | Batches 1 (build) + 2 (tripwire) PASSED. Batch 3 (PPL) running. Batches 4-5 pending. |
| 5 — Artifacts | this bundle + handoff completion + branch push | Pending Phase 4 completion. |

## Bundle layout

```
2026-04-30-v5-cleanup-audit/
├── README.md                                # this file
├── system-state.txt                          # pre-validation host state
├── phase4-validation-gates/
│   ├── run_validation.sh                    # the runner script that produced batch{2,3,4}-*.log
│   ├── batch2-tripwire.log                  # Coder-30B Q4_K_M tg32 r=5 canonical
│   ├── batch3-ppl-{coder30,q8,reap,gemma}.log
│   ├── batch4-bench-{coder30,q8,reap,gemma}.log
│   └── batch5-note.md                       # per-role smoke deferred to orchestrator integration
└── phase5-artifacts/
    ├── decision.md                           # final go/no-go for v5 push
    └── branch-log.txt                        # git log production-consolidated-v4..production-consolidated-v5
```

## Cherry-pick scope summary

50 commits in chronological order:
- CPU1 stack: 12 commits (will be partially soft-stripped post-cherry-pick — NUMA_WEIGHTS family)
- op-fusion + reverts: 4 commits (canceling, no net effect)
- CPU2 Q8_0 ukernel + follow-up: 2 commits
- gated_delta_net + rms_norm: 2 commits (will be stripped post-cherry-pick)
- CPU15 Phase 1+2 superseded: 3 commits (will be stripped post-cherry-pick)
- CPU15 Phase 3.2 inter-process EP: 13 commits (KEEP)
- CPU2 mbind kill-switch: 1 commit
- CPU1 P1.3 fix per-region mbind: 1 commit (part of NUMA_WEIGHTS, soft-stripped)
- CPU2 Q6_K ukernel: 3 commits
- CPU22 work-stealing: 1 commit (stripped post-cherry-pick)
- MoE-Spec budget: 1 commit
- slot-promotion v0+v1: 6 commits
- CPU4 op-coalesced barriers: 1 commit

5 NUMA_MIRROR commits skipped (decisive negative on single-socket NPS4, CPU25 closure).

## Strip + refactor commits (8 forward-style on top of cherry-picks)

| SHA | Subject |
|---|---|
| `f6418b48a` | strip CPU22 work-stealing prototype (closure-via-test) |
| `d47ce5660` | strip GGML_RMS_NORM_PARALLEL path (net-negative) |
| `af42e982b` | strip GGML_GDN_K_PER_HEAD path (no current effect) |
| `bbe38a683` | strip CPU15 Phase 1+2 dead paths in mul_mat_id |
| `b43dda32f` | strip CPU15 Phase 2 anon-copies producer (loader) |
| `f314f1b04` | strip GGML_NUMA_WEIGHTS family activations (mmap hard, loader 3× soft) |
| `47991c6b2` | gate EP informational logs behind GGML_EP_VERBOSE |
| `734d011e0` | name MUL_MAT_BLOCK constant (refactor) |

## Cross-references

- Handoff: `handoffs/active/v5-push-cleanup-audit.md` (will move to `handoffs/completed/` post-validation)
- Inventory: `handoffs/active/cpu-kernel-env-flags-inventory.md` (updated Phase 0 SHA reconciliation)
- Deployment-draft: `handoffs/active/model-registry-v5-deployment-draft.yaml`
- Research deep-dive (Q3 output): `research/numa-weights-deep-dive.md`
