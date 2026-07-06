#!/bin/bash
set -euo pipefail

# E2 eval-driver A/B. Run arms sequentially in a clean/quiesced window.

# arm: batch_np8_single_full_instance
# blocked: host-health preconditions failed; rerun after clean host-health or pass --allow-host-health-warning for explicitly non-decision-grade scout data
# (
#   cd /mnt/raid0/llm/epyc-inference-research
#   uv run --extra benchmark python scripts/benchmark/server_np_sweep.py --run-id e2-pbench3-plan-20260706T032647Z-batch-np8 --output-root /mnt/raid0/llm/epyc-inference-research/data/batched_decode/e2-pbench3-plan-20260706T032647Z/serving --model-key qwen36_q8_0 --np-levels 8 --prompt-limit 43 --prompt-seed 42 --tier 1 --n-predict 256 --port-base 18070
# )

# arm: current_three_concurrent_quarters
# blocked: host-health preconditions failed; rerun after clean host-health or pass --allow-host-health-warning for explicitly non-decision-grade scout data
# (
#   cd /mnt/raid0/llm/epyc-orchestrator
#   AUTOPILOT_EVAL_CONCURRENCY=3 uv run python scripts/autopilot/core_v2_calibrate.py --calibration-id e2-pbench3-plan-20260706T032647Z-current-quarters --out-jsonl /mnt/raid0/llm/epyc-inference-research/data/batched_decode/e2-pbench3-plan-20260706T032647Z/current_quarters.jsonl --n 43 --repeats 1 --seed 42 --trial-id-base 920000 --overwrite
# )
