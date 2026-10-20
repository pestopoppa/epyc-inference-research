#!/bin/bash
# job1: materialise clips -> self-export streaming encoders -> NeMo CPU references. One build claim.
D=/mnt/raid0/llm/tmp/cs-46-20261020
W=/mnt/raid0/llm/worktrees/cs-46-parakeet-eou-research-20261020
S=$W/scripts/speech/parakeet_eou
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 HF_HOME=$D/hf HF_HUB_OFFLINE=1
mkdir -p $D/out $D/logs
echo "[job1] start $(date -u +%FT%TZ)"
cd $W/scripts/voice/turn_taking
$D/venv-arm/bin/python -I tt_materialize.py --corpus $W/data/turn-taking-cs45-v1 --out $D/clips > $D/logs/materialize.log 2>&1; echo "[job1] materialize rc=$?"
cd $D
$D/venv-nemo/bin/python -I $S/export_streaming_models.py --nemo $D/nemo/parakeet_realtime_eou_120m-v1.nemo --out $D/export --threads 8 > $D/logs/export.log 2>&1; echo "[job1] export rc=$?"
for cm in 160 80; do for rs in full none; do
  $D/venv-nemo/bin/python -I $S/nemo_ref.py --nemo $D/nemo/parakeet_realtime_eou_120m-v1.nemo --clips $D/clips.jsonl --out $D/nemo_ref/c${cm}_${rs} --chunk-ms $cm --reset $rs --threads 8 > $D/logs/nemo_ref_c${cm}_${rs}.log 2>&1; echo "[job1] nemo_ref $cm $rs rc=$?"
done; done
echo "[job1] done $(date -u +%FT%TZ)"
