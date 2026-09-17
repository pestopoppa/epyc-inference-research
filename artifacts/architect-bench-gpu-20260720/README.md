# architect-bench GPU draft-depth sweep — 2026-07-20

Measurement evidence made durable on **2026-09-15** (NIB2-73a). The master registry cited
three `results.jsonl` files here that existed only as UNTRACKED files in the shared clone
`/mnt/raid0/llm/epyc-inference-research`. The consequence was not hypothetical: the
evidence gate passed in that one checkout and reported `MISSING` in every worktree, and a
single `git clean` would have taken the evidence behind three ratified acceleration
claims with it. Copied byte-for-byte within the same clone (sha256-verified both ends);
the untracked originals were left in place, as the 2026-08-02 migration did.

| | |
|---|---|
| origin | `/mnt/raid0/llm/epyc-inference-research/artifacts/architect-bench-gpu-20260720/` (untracked) |
| measured (file mtimes, UTC) | 2026-07-20 18:06 – 18:13 |
| made durable | 2026-09-15 |
| carried | 3 files, 9,656 bytes |

## What was measured

One campaign, three arms, one variable: **draft depth** (`--spec-draft-n-max`) on MI210
under `production-consolidated-v7`, at production sampling (temp 0.6, top_p 0.95, top_k
20, seed 42), 512-token generations, best of n=2 repetitions, f16 KV, `-c 32768`,
`-b/-ub 2048`, `-ngl all -fa on`. Each record in a `results.jsonl` is one config:
`label`, `best_wall_tok_s`, a `runs[]` array (`elapsed_s`, `completion_tokens`,
`wall_tok_s`, `srv_predicted_per_second`, `srv_prompt_per_second`, `finish_reason`,
`sane`, and a `tail` excerpt of the generation for the coherence check) and `vram_pct`.

All three arms are **observation-grade (pre-`P-GPU-1`)** and the registry says so at each
citation. Nothing here is a ratified serving number.

## Registry claims this backs

`orchestration/model_registry.yaml` — these citations resolve to this directory. The YAML
key path is the stable reference; line numbers are as of 2026-09-15.

- **L3593** &nbsp;`roles.qwen35_122b_iq2m.acceleration.gpu_spec_depth_sweep_observation`
  → `sweep_a1/results.jsonl`. Qwen3.5-122B-IQ2_M: **+51% at n-max 2**. This arm exists to
  QUALIFY the sibling `native_mtp_observation`, which had concluded "keep this candidate
  no-spec by default" from a 0.61x MTP result measured without controlling draft depth.
  Draft depth, not MTP, was the uncontrolled variable — so the earlier conclusion was an
  artifact of the setup, and this file is what shows that.
- **L9960** &nbsp;`roles.qwen36_27b_mtp_q8_local.acceleration.gpu_spec_depth_sweep_observation`
  → `sweep_a3/results.jsonl`. Qwen3.6-27B: no-spec 29.2 → n-max 2 38.8 → **n-max 4 53.1
  t/s (1.82x)**, the largest MTP win of the three arms.
- **L10163** &nbsp;`roles.qwen36_35b_a3b_mtp_q8_local.acceleration.gpu_spec_depth_sweep_observation`
  → `sweep_a4/results.jsonl`. Qwen3.6-35B-A3B (`10098 6ad45fa3f`, build-hip): no-spec
  84.8 → n-max 2 94.5 → **n-max 4 104.0 t/s (1.23x)**; fastest decode of the three arms,
  as expected for 3B active parameters.

The cross-arm finding is the one worth keeping: the optimum is n-max 2 for the 122B-IQ2
and n-max 4 for both Qwen3.6 models, so **draft depth must be tuned per model, never
inherited**.

## Reading these files — two defects in the substrate, recorded not fixed

1. **`sweep_a3` and `sweep_a4` each begin with three `SERVER_DIED` stubs**
   (`{"arm":"a3","label":"specnone","error":"SERVER_DIED"}`) followed by three successful
   reruns. The live numbers are lines 4-6. The stubs are kept because deleting a failed
   attempt from a results file is how a campaign starts lying about its own yield.
2. **`sweep_a1` line 6 is not strictly valid JSON.** The `tail` excerpt contains raw
   LaTeX (`\hline`, `\end{array}`), i.e. unescaped `\h` / `\e`, so `json.loads` raises
   `Invalid \escape: line 1 column 301`. Lines 1-5 and all of a3/a4 parse cleanly. A
   consumer needs a lenient parser. The bytes are carried AS MEASURED — repairing the
   escaping would change a file whose sha256 is the thing being attested.
3. `vram_pct` is present on `sweep_a1` records 1-4 and absent on `mtp1`/`mtp3`. Absent,
   not zero; do not read the gap as a measurement.

## What was NOT carried

The three `sweep_a*/` directories also hold per-config subdirectories (53,090 bytes
total across 51 files: per-config server logs and raw response dumps). Nothing cites
them, they are substrate rather than result, and the distilled per-config rows in
`results.jsonl` are what the registry claims rest on. They remain untracked at the origin
path above.

## Not to be confused with the v9 mmlu_pro runs already in this directory

This directory ALSO holds 46 files that were already tracked before this migration:
`runs/mmlu_pro/A{1,3,4}_*_v9/`, the `A*_v9.claim_{open,released}.json` device-claim
receipts, `device_claim_journal.jsonl` and `questions_mmlu_pro.json`. Those are a
different capture — accuracy on mmlu_pro under **v9** — and they are not what the three
citations above point at. The three `sweep_a*/results.jsonl` files are the 2026-07-20
**v7 draft-depth throughput** sweep, and they were the untracked part.

## Integrity

`SHA256SUMS` seals **49 files** — the 3 carried `results.jsonl` plus the 46 that were
already tracked, so it is a campaign-wide integrity record rather than a receipt for one
commit. `README.md` is deliberately not in it (documentation is not evidence, and hashing
it would make every doc edit break the seal), following the 2026-08-02 precedent in
`data/paddleocr_vl_receipt_extract_20260717T194415Z/`. The three carried files were hashed
after the copy and compared against the originals; all verified `OK`. Verify with:

```bash
cd /mnt/raid0/llm/epyc-inference-research && sha256sum -c artifacts/architect-bench-gpu-20260720/SHA256SUMS
```
