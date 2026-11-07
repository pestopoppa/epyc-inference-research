# Hypothesis-portfolio authority re-seal review — 2026-11-07

Scope: changes to `discovery_hypothesis_portfolio_v2.json` and `hypothesis_portfolio.py` since the
last pin update `5ae54e0ae`: `f8167767a` (MI210 port fast priors) and `a4e2201bf` (isa_schedule).
Procedure (as in 5ae54e0ae): review diff, recompute file sha256 + semantic sha (`load().sha256`) +
contract sha (`HYPOTHESIS_PORTFOLIO_V2.md`), update the pins in `controller/discovery_deployment_factory.py`
and the semantic-sha assertion in `controller/test_discovery_deployment_factory.py`.

## Verdict: CLEAN — re-sealed

| Check | Result |
|---|---|
| Rows dropped | none. hypotheses 21->21, frames 4->4, evidence 29->30 (+`ev-mi210-asm-port-drops-20261106`), DNR 22->23 (+`dnr-gfx90a-launch-removal-fusion-under-hip-graphs`) |
| Evidence carrier resolves | yes: `evidence/mi210-asm-port-20261106/mi210-port-drops-20261106.md` sha256 `f2ad7348…b2ed` matches; `verify_evidence_files` passes |
| Fingerprint stability | all 43 pre-existing mechanism blocks: old vs new `mechanism_fingerprint` identical, facets and stored fingerprints unchanged. a4e2201bf only adds a validation branch for `change_class == "isa_schedule"` (no existing row uses it); hash input unchanged. New DNR stored fingerprint `8eebb0d3…` equals recomputation |
| H4 numbers | program.md:730 + gate `ak-asm-h4h6-gate-20261105/out/RESULT.json`: geomean c6/new 0.482, e2e pp2048 587.46 vs 834.91 (H4-off arm; C6 833.96). Matches "0.48x, 587 vs 835" |
| H9 numbers | program.md:731 + `ak-asm-h9-gate-20261106/out/RESULT.json`: 0.524. Matches "0.52x" |
| H9B | program.md:732 + `ak-asm-h9b-gate-20261106/out/RESULT.json`: 0.642. Matches |
| H14 | `ak-asm-h14h15-gate-20261106/out/RESULT.json` H14 DROP "e2e on/off 1.0006 (f16) / 0.992 (q8_0)"; op 1.13 from r3/r4 (1.1315 / 1.1346). Matches |
| Contract | `HYPOTHESIS_PORTFOLIO_V2.md` sha unchanged (`96f20773…`), pin kept |

## Minor discrepancy (non-blocking, recorded)

- **H6 "1.017x"** (DNR falsifier_result, carrier, progress 2026-11-06, handoff autokernel-loop.md:5622)
  is not reproduced by the only H6 RESULT.json on disk: `ak-asm-h4h6-gate-20261105/out/RESULT.json`
  reports `h6_verdict: INCONCLUSIVE`, one cell `h6off_over_new 0.9997`. The 1.017 figure has no
  surviving raw packet. Not blocking: both values are below the 1.10 bar, and the DNR is a refusal,
  so the weaker on-disk number only strengthens it. Follow-up for the carrier owner: replace 1.017
  with the on-disk value or cite its source.
- H4 "835" is the same-binary H4-off arm; C6 itself is 834. Immaterial.
- Carrier `temporal_status: current_v9` although measurements are against C6 (v10 era); the field's
  enum is the owner's to update.

## New pins
- file `276623643c9b540970be73067797e217183a55212e57ffed4e9f518aec5c6f1f`
- semantic `9dcb51a8e3d2c037eaa6ffbf10248b31e625e67cf3f498ff71ae5c8d70c2cb43`
- contract `96f207733e5fc27a722763cf1b3c542f327eb70d41e04b9948aaec086b3facd4` (unchanged)
