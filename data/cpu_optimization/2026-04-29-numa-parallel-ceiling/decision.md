# 4-NUMA Aggregate Ceiling — Decision: Phase 2 GO

## Verdict

**GO** for slot-promotion Phase 2 (NUMA-parallel candidate verify). Aggregate ceiling 6.10× single-instance; gate ≥1.3× has 4.7× headroom for orchestration overhead.

## Phase 2 implementation effort (revised honest estimate)

Original Phase 0 estimate of ~50-100 LOC was for the seq_cp wiring (which turned out to already exist). The actually-new work is:
- Multi-context model loading per request (K contexts pinned to K NUMA quarters): ~150-250 LOC in `tools/server/server-context.cpp` + threadpool refactor
- Coordinated accept/reject across K candidate slots: ~100-200 LOC in spec-dec verifier path
- Testing + integration: ~1 week
- Total: ~300-500 LOC, ~1-2 weeks wall-clock

Multi-day implementation; queued for next focused session.

## Bonus actionable finding (independent of Phase 2)

1 NUMA quarter at 24t runs ~1.7× faster than full machine at 96t on Qwen3.6-35B-A3B Q8. Existing 4×48t orchestrator (per `numa-orchestrator-deployment.md`) may be leaving throughput on the table for hybrid Delta Net frontdoor.

This warrants a separate orchestrator probe: 4×24t aggregate vs 4×48t aggregate on Q8 frontdoor role. NOT done here. Separately actionable.

## Phase 2 deferral rationale

Phase 1.0 GATE MET + Phase 2 ceiling probe MET = strong GO verdict. Implementation is multi-day so deferred to dedicated session. The 6.10× ceiling means the question is no longer "does Phase 2 have gain?" (yes) but "how cleanly does the orchestration converge?" — that's an engineering question, not a research one.
