# AMENDMENT -- V11-086 v2 integrity gate: FP32 `use_ref` oracle (2026-10-23)
Addendum to `predeclared/086v2/PREDECLARED_086v2.md` (sha 48897be2, left unchanged and pinned). Written 2026-10-23 by 3c4ca599
BEFORE any rerun. Authority: operator chose **option B** of `/mnt/raid0/llm/tmp/v11-runs/086v2/longfa/DECISION.md` on 2026-10-23.

## What changes
1. The test-backend-ops oracle (`use_ref` CPU reference backend) accumulates F16-V FLASH_ATTN_EXT in FP32 whatever the GGML_FA_VKQ_*
   knobs say. Code: llama.cpp `experimental/v11-086-fp32-oracle-20261023` @ d6cabef11 (on B 8110933ba) and
   `experimental/v11-086-fp32-oracle-on-c-20261023` @ df3e0f8cf (on C f0c96dc1f). Production (non-reference) execution is unchanged.
2. Tolerance stays **5e-4** (NMSE/ERR as test-backend-ops reports). No other number in the predeclared file changes.
3. The integrity inputs "tbo FA 100%" are re-measured against the new oracle (fa_unset_bit_identical, finite_all_dumps, knob proof and
   linkage are oracle-independent and are NOT re-measured).

## Predeclared rerun (jobs/v11_086_fp32_oracle_job.sh; scorer v11drv/fa086v2_oracle_rerun.py)
- PHASE 1: standard corpus (5207 cases) on C+oracle. The predicted near-kv16k cases (~3.6e-4) must stay < 5e-4 => 5207/5207.
  Any failure stops the rerun (PHASE1_FAIL) and returns to the operator: the oracle change would then break the champion's standard gate.
- PHASE 2: standard corpus on Bunset, B REG=1, B REG=auto; the 35-case long set (`cpu_fa_longctx_v1`, filter nr23=[(12,1|64,1)],kv=...)
  on C, Bunset, B REG=1, B REG=auto.
- PREDICTION (declared before any run): all standard 100%. Long set: B REG=1 and REG=auto 35/35. C and Bunset 27/35, failing exactly
  the 8 D256 nb>=2 cells (kv 65536/131072) -- C's FP16 hybrid vs an FP32 oracle, the mirror of REG vs the FP16 oracle (same 7e-4..3e-3
  band); the 4 nb=1 cells pass on every arm (split-KV chains are short). If C passes all 35, the FP16-accumulation explanation of the
  8 cells is wrong; if the nb=1 cells still fail on C, the split-KV path is the outlier and that is a separate champion finding.

## Consequence stated in advance (clause A3 -- operator to confirm; main proceeds under this reading)
The unamended rule ("tbo FA 100% on C, Bunset, B1, Bauto") would HOLD again on the predicted C failures, which are a property of the
champion's FP16 accumulation and not a defect of the candidate. So the amended integrity input is:
- A1 standard corpus 100% on all four arms;
- A2 long set 100% on B REG=1 and B REG=auto (the candidate);
- A3 long set on C and Bunset: failing sets IDENTICAL (identity control). C's failures are recorded as a known champion property
  (FP16 accumulation over long KV), not counted against the candidate.
`tbo_fa_100 (amended) = A1 and A2 and A3`; the strict letter (all 8 runs 100%) is kept alongside as `tbo_fa_100_letter`. The decision
rule (hold / default_on_VKQ_REG_auto / fold_opt_in_default_off) is otherwise unchanged and still reads `tbo_fa_100`.
If the operator strikes A3, the strict letter governs and a C long-set failure keeps V11-086 on HOLD.

## Install step (after the rerun, main)
Move `correctness/INTEGRITY.json` to `INTEGRITY.json.bak-pre-fp32-oracle`, write `ORACLE_RERUN.json["integrity_amended"]` as
`correctness/INTEGRITY.json`, then re-arm the timing poller. Pin this file (`v11_driver_086v2.py --pin` pins PREDECLARED files only;
add its sha to `predeclared/PINNED.sha256` by hand).
