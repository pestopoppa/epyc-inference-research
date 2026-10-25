# OPERATOR DECISION -- V11-086 v2 clause A3 accepted; REG passes (2026-10-23)
Addendum to `ORACLE_AMENDMENT_20261023.md` (sha f0ef6c54, pinned). Written 2026-10-23 by 3c4ca599 on the operator's instruction
(session 3c4ca599, decision ~16:55Z). Authority: operator.

## The decision (verbatim intent)
Operator, 2026-10-23 ~16:55Z, on V11-086: **"A3 as accepted: REG passes"**, and **keep investigating the D512 nb>=2 cells**.
Accepted wording of A3: **"REG is judged only on cells outside the C/B-unset common fail set"**. The operator ruled that this reading
governs the integrity input. Under it REG adds no failures relative to the champion, so integrity passes.

## Evidence (FP32 `use_ref` oracle rerun; /mnt/raid0/llm/tmp/v11-runs/086v2/fp32oracle/ORACLE_RERUN.json, written 2026-10-23T16:51:48Z;
scorer v11drv/fa086v2_oracle_rerun.py)
- Standard corpus: 5207/5207 on all four arms (C, Bunset, B1, Bauto).  A1 = TRUE.
- Long set `cpu_fa_longctx_v1` (35 cases):
  - C 19/35 and Bunset 19/35, IDENTICAL fail sets of 16 keys: hsv=256 kv=65536/131072 nb=2..5 (8 cells) and hsv=512 kv=32768/65536 nb=2..5
    (8 cells). Max err C 2.967e-3, Bunset 2.892e-3. A3-identity = TRUE.
  - B1 27/35 and Bauto 27/35, failing exactly the 8 hsv=512 kv=32768/65536 nb=2..5 keys (all in C's set; subset check run: TRUE for both).
    Max err B1 1.415e-3, Bauto 1.407e-3.
  - REG fixed all 8 D256 cells; the 8 D512 cells fail on every arm.
- As pinned, A2 (REG 35/35) is FALSE, so tbo_fa_100 (amended, A1 and A2 and A3) as pinned is FALSE, and the strict letter is FALSE.

## Ruling recorded
**A2 superseded by the operator's A3 wording: candidate passes iff its long-set fail set is a subset of the C-and-Bunset common fail set
AND A1 AND A3-identity.**
Applied: B1 fail set (8) and Bauto fail set (8) are each a subset of the common set (16); A1 TRUE; A3-identity TRUE =>
**tbo_fa_100 (operator rule) = TRUE.** Recorded alongside, not erased: `tbo_fa_100_letter` = FALSE; `tbo_fa_100_A2_as_pinned` = FALSE.
The decision rule (hold / default_on_VKQ_REG_auto / fold_opt_in_default_off) is otherwise unchanged and still reads `tbo_fa_100`.
This ruling does NOT claim the D512 cells are correct: they are a measured property of the champion AND the candidate vs the FP32 oracle
(V11-086-D512 investigates).

## Failed prediction (recorded honestly)
The amendment predicted, before any run: B REG=1/auto 35/35; C and Bunset 27/35 failing exactly the 8 D256 nb>=2 cells. Actual:
- B REG 35/35: FALSE (27/35; 8 D512 cells fail).
- C 27/35 with only D256 failing: FALSE (C fails 16 = 8 D256 + 8 D512). The amendment's explanation (FP16 accumulation over long KV is the
  whole story) covers D256 but does not explain D512, where REG does not help.
- No nb=1 cell failed on any arm: as predicted.  C == Bunset fail sets: as predicted.
Consequence: the "known champion property" is broader than predicted (16 cells, not 8); KP-1 is updated to the measured 16 cells.
