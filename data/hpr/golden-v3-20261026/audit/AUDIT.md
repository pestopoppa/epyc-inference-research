# HPR golden v3 result audit — PASS (bounded instrument scope)

Read-only audit on 2026-10-26. Evidence base: `/mnt/raid0/llm/tmp/codex-ni-20261024-hpr-recovery/run-golden-v3/live/` (abbreviated LIVE below). No inference, test, process restart, source mutation, ledger write or shared-document edit performed.

The completed native job is fresh: request timestamps span 03:32:52–05:40:49Z on October 26; COMPLETE and RESULT-READY were written 05:40:57Z; official `.ended` and `.complete` record rc=0 at 05:40:59Z. GOLDEN.EXIT is exit=0. The official grant is now absent. The used grant was CPUs24–47, 03:30–09:35Z, digest19391106…; neither restart nor new execution is required.

## What passed

- `GOLDEN_RESULT.json`: schema `epyc.hpr.golden_parity.v1`, status PASS, exact declared scope. Four fresh paired families: 8k/dense, 8k/tail_junction, 64k/dense, 64k/coarse. Each has cold, matched_extension, divergent_edit on chat and native endpoints: **24 fresh request captures, 12 paired comparisons, 8 fresh servers**.
- Independently queried all 12 pairs: exact equality of prompt IDs, generated IDs, retained boundary, prompt/cache counts, checkpoint/pinned positions, restored count and full-reprocess flag. All recorded failures, validation_problems, simulator mismatches and incomplete flags are empty/false. All 24 request rows have HTTP200 and no raw error.
- All24 raw captures contain16 sampled-token debug records and exactly one untruncated release. All12 native captures have16 partial IDs, one final event, empty native anomalies, predicted_n/tokens_predicted16, evaluated count equal prompt length, stop_type=limit, truncated=false, empty final token array. Source `golden_checks.py` enforces isolated slot/task and sequential n_decoded1..16; native SSE/trace/boundary agreement is checked before PASS.
- Required controls independently recomputed from summaries: both cold caches zero; matched LCP strictly enters retained generated suffix; divergent LCP precedes prior retained end; every tail_junction native request pins the explicit final-user position. No additional ad hoc negative-control requirement was invented.
- Before/after body affinity is24–47 and launch receipts agree. All7,626 during-request samples across8 endpoints have nonempty affinity subsets of24–47, active-request=true, and libraries mapped inside the exact experimental store. Per-endpoint counts1835,1926,1844,1805,56,53,53,54 exceed the required two samples. A sampled main thread can be pinned to CPU24; this is a valid subset and not an escape.
- Eight peak-memory receipts say RECORDED. Eight cleanup receipts say REAPED, rc0 after TERM. A read-only ps of those eight captured PIDs plus supplied ended watcher/poller PIDs found none present. No process-name search/signalling was used.

## Pins and provenance

`OWNER-PINS.SHA256SUMS` and the entire `SOURCE-NEXT.SHA256SUMS` closure check successfully now. The run log also records successful source checks after completion. Driver helpers inspected from research git object c4c2a3638cf6a62c823920c3d9d271e6aec8f911 match the preregistered16-token/temperature0/seed42/cache/slot/stream request contract and junction conversion. Exact experimental source80a09c6ad86b3c4fd6acdea7d2cb65de8d4f34d8 still resolves at HEAD with clean tracked state; its server-context.cpp logs the sampled token before return. Build provenance explicitly identifies an experimental CPU image, not production/champion. Its distinct build-info ancestor dd5bb0d3b is documented rather than silently treated as the source tip.

The pinned launcher is1442c95d…; cheap preflight verified29 store entries plus the DEBUG format in implementation library2fe34db8…. Linkage returncode0 and every launch LD_LIBRARY_PATH target that experimental store. The unchanged audited `verify_store_then_model` hashes all model bytes and refuses mismatch before any server starts, binding this completed execution to model SHA256c1283d8b80c3e38b2735ddbc9766d3b3126f44d6c484be419d4e101d09a76131. This review did not reread the37.8GB model. There is no evidence of production-kernel mutation in this job: launcher/source paths, subprocess actions and write destinations are experimental/scratch; no build/deploy action exists in the golden body.

## Accepted claim and remaining gates

Accept only prospective **instrument equivalence for these fixed four families and three16-token probes**, on this pinned CPU experimental binary/model/template/settings. `existing_family_observation_count=72` is the existing-reader projection count from24 newly emitted request rows; it is not72 probes,72 fresh model comparisons, a new family/ladder, or permission to regrade historical data. The artifact scope and every request row retain native-contract UNVERIFIED.

No speed/model-quality/general stop/EOS/sampler claim, production promotion, source publication, parent HPR closure or historical regrade follows. Remaining named gates: MAIN/owner acceptance of this bounded result; explicit review/decision for any native-contract admission; separately authorized source merge/publication; a later preregistered measurement before performance or broader policy claims. Earlier failed-v1/v2 artifacts and unknown original peakRSS/64k coarse timing remain historical limitations, not gaps in this completed fixed control.

## Evidence hashes (SHA256)

| LIVE-relative path | SHA256 |
|---|---|
| golden-out-v3/GOLDEN_RESULT.json | 3e0f03f0697f678364a7b8c515158ab377545c7e8ba676b0129357acd15c1ec1 |
| golden-out-v3/requests.jsonl | ebea08d5609bde74d7627815d76e556afdb76dc3e89bbd68985587a7e7ba567c |
| PREDECLARATION.md | c08e7cf0929248224d2e90585af90b5a14f60ce535b1687e6efd23cb7720d2b0 |
| SOURCE-NEXT.SHA256SUMS | 4e01fb4452d8a25906fa77a220d7d038dd5059d7a8f031be0be822da93fa641c |
| OWNER-PINS.json | fff61ce908836a572692b826a4af894e7bf4c860e4f82d04436f3a4fd5f3a48d |
| GOLDEN.grant.used.txt | 1939110688f4228dd19aade53210ecfb70fde373cfeb7cba6044f582067cdf87 |
| GOLDEN.AFFINITY.before.json | 82566be706496f79c3b4524234b9fe39da32afddb017bb30cac826d61596ef92 |
| GOLDEN.AFFINITY.after.json | 630639db7b9854ccffe913babda3c07e023afb00df110dcc6446a56c2ed5df07 |

Detailed native evidence remains in each `golden-out-v3/<unit>-<arm>-<endpoint>/`: launch.json, server.log, three *.raw.json, during_request_samples.json, endpoint_ownership.json, peak_memory.json, cleanup.json. This audit preserves those files unchanged and avoids dumping token arrays or raw wire data.
