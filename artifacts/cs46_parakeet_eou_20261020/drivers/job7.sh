#!/bin/bash
# job7: deterministic (eval-frontend) NeMo refs, 4 combos, one claim
D=/mnt/raid0/llm/tmp/cs-46-20261020
S=/mnt/raid0/llm/worktrees/cs-46-parakeet-eou-research-20261020/scripts/speech/parakeet_eou
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8
echo "[job7] start $(date -u +%FT%TZ)"
for cm in 160 80; do for rs in full none; do
  $D/venv-nemo/bin/python -I $S/nemo_ref.py --nemo $D/nemo/parakeet_realtime_eou_120m-v1.nemo --clips $D/clips.jsonl --out $D/nemo_ref/c${cm}_${rs}_eval --chunk-ms $cm --reset $rs --threads 8 --frontend-mode eval > $D/logs/nemo_ref_c${cm}_${rs}_eval.log 2>&1; echo "[job7] $cm $rs rc=$?"
done; done
# also re-run shipped mode for 160/none to measure NeMo's own run-to-run variability (seeded)
$D/venv-nemo/bin/python -I $S/nemo_ref.py --nemo $D/nemo/parakeet_realtime_eou_120m-v1.nemo --clips $D/clips.jsonl --out $D/nemo_ref/c160_none_shipped_rerun --chunk-ms 160 --reset none --threads 8 --frontend-mode shipped > $D/logs/nemo_ref_rerun.log 2>&1; echo "[job7] rerun rc=$?"
echo "[job7] done $(date -u +%FT%TZ)"
