# Deferred Gold Validation Workflow

This workflow is deliberately not executed while the E8 baseline reseed requires
clean CPU measurement windows. It applies to the immutable 180-ID candidate list
in `powered_160_candidate_manifest.json`.

## Gold validation

From `/mnt/raid0/llm/epyc-inference-research`, after the E8 owner releases the
clean-CPU boundary, run the official gold source on the full candidate list:

```bash
set -euo pipefail
BASE=/mnt/raid0/llm/epyc-inference-research/artifacts/architect-laguna-iq2-v8-20260726/a3-a4-swe-confirmation
IDS=$(tr '\n' ' ' < "$BASE/powered_160_candidate_manifest.ids.txt")
cd "$BASE"
/mnt/raid0/llm/epyc-inference-research/.venv-swebench/bin/python -m swebench.harness.run_evaluation \
  --dataset_name princeton-nlp/SWE-bench_Verified \
  --predictions_path gold \
  --instance_ids $IDS \
  --max_workers 8 \
  --cache_level env \
  --run_id a3-a4-powered-gold-v8-20260726 \
  --report_dir "$BASE/gold_validation"
```

The harness report is the authority. It must show at least 160 resolved gold IDs,
and prove the complete 180-ID run: `submitted_ids` must exactly equal the manifest
candidate set; total/submitted counts must both be 180; all completed/resolved/
unresolved/empty list and count partitions must be internally consistent; and no
incomplete/error IDs or outstanding containers may remain. The summary JSON is
written in the current directory at the exact path
`$BASE/gold.a3-a4-powered-gold-v8-20260726.json`; `--report_dir` is only the
per-instance-log location. Preserve the full report and select the first 160
`resolved_ids` in immutable manifest order. Do not rank, filter, or replace IDs
based on model output.

Run the checked acceptance validator immediately after the harness exits:

```bash
python3 "$BASE/validate_powered_gold_report.py" \
  --manifest "$BASE/powered_160_candidate_manifest.json" \
  --gold-report "$BASE/gold.a3-a4-powered-gold-v8-20260726.json" \
  --accepted-ids-out "$BASE/accepted_160.ids.txt" \
  --summary-out "$BASE/gold_acceptance.json"
```

## Prompt materialization

The historical `build_swebench_prompts.py` intentionally hard-codes the old
40-item gold report. Do not overwrite its historical inputs or questions file.
Create a new, parameterized copy in this campaign artifact (or extend the utility
with explicit input/output CLI paths and tests), then materialize prompts only
from the accepted 160-ID list. The implementation must retain the historical
oracle mode: base-commit file content, 120-line hunk windows, identical
SEARCH/REPLACE instructions, and no test modification permission.

Record the accepted-ID list, prompt JSON SHA-256, prompt-character distribution,
model SHA-256s, raw JSONL, conversion summaries, and official reports. Run A3 and
A4 one GPU sidecar at a time with the identical v8 HIP configuration:
`ROCm0`, `-ngl all`, `-fa on`, f16 K/V, `-c 49152`, MTP draft `n=4`, reasoning
off, seed 42, temperature 0.6, top-p 0.95, top-k 20, and 3072 completion tokens.

For each terminal converted arm, use the exact accepted list for the official
patch evaluation. These commands are deliberately deferred with gold validation:

```bash
IDS=$(tr '\n' ' ' < "$BASE/accepted_160.ids.txt")
cd "$BASE/A3"
/mnt/raid0/llm/epyc-inference-research/.venv-swebench/bin/python -m swebench.harness.run_evaluation \
  --dataset_name princeton-nlp/SWE-bench_Verified \
  --predictions_path "$BASE/A3/predictions.json" \
  --instance_ids $IDS \
  --max_workers 8 \
  --cache_level env \
  --run_id a3-powered-swe-v8-20260726 \
  --report_dir "$BASE/A3/official_report"

cd "$BASE/A4"
/mnt/raid0/llm/epyc-inference-research/.venv-swebench/bin/python -m swebench.harness.run_evaluation \
  --dataset_name princeton-nlp/SWE-bench_Verified \
  --predictions_path "$BASE/A4/predictions.json" \
  --instance_ids $IDS \
  --max_workers 8 \
  --cache_level env \
  --run_id a4-powered-swe-v8-20260726 \
  --report_dir "$BASE/A4/official_report"
```

## Final statistic

Score each arm over the full accepted 160-item denominator, treating empty
patches as failures. Use the official FAIL_TO_PASS result per instance and report
paired discordants plus two-sided exact McNemar p. The expanded run is not
decision-grade until both arms share exactly the same accepted ID list and no
gold/model-evaluation errors are hidden by denominator shrinkage.
