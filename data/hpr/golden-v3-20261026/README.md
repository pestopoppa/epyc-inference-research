# HPR golden v3 instrument-equivalence evidence (2026-10-26)

Durable copy of the MAIN-accepted HPR golden v3 run (q1 grant CPUs 24-47, 03:30-09:35Z; native COMPLETE 05:40:57Z).
4 families (8k dense, 8k tail_junction, 64k dense, 64k coarse) x 3 probes (cold, matched_extension, divergent_edit)
x 2 endpoints (chat, native) = 24 fresh captures / 12 exact comparisons; schema epyc.hpr.golden_parity.v1 PASS.

Scope: accepted ONLY as prospective instrument equivalence on the pinned experimental CPU model/settings.
Native contract UNVERIFIED. No historical regrade, no performance claim, no parent HPR closure; 64k holdout stays open.

- `GOLDEN_RESULT.json` sha256 3e0f03f0697f678364a7b8c515158ab377545c7e8ba676b0129357acd15c1ec1
- `audit/` accepted checkpoint, AUDIT.md (sha256 64b436763804175006fd8da78671c31e3fb714c19ef6e9985e1a08d3d517f0c2), owner closeout draft, native SHA256SUMS
- `<family>-<endpoint>/` raw captures, server.log, during_request_samples.json, cleanup.json
- `SHA256SUMS` covers every file in this directory

Native source: `/mnt/raid0/llm/tmp/codex-ni-20261024-hpr-recovery/run-golden-v3/live/golden-out-v3/` (tmp; not durable).

## Large native captures

The four raw captures previously omitted because each exceeded 5 MB are now stored losslessly in [large-originals/](large-originals/README.md). Its native JSON manifest records source before/after hashes and sizes, compressed hashes and sizes, full decompressed verification, and actual archive timestamps. The existing 44 native files remain unchanged.

This is preservation of the accepted 24 captures / 12 comparisons. The 72 reader projections are not 72 probes. Native contract remains UNVERIFIED; no historical regrade, parent HPR closure, new measurement class, warrant, or grade.
