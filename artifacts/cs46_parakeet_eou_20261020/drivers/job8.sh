#!/bin/bash
D=/mnt/raid0/llm/tmp/cs-46-20261020
S=/mnt/raid0/llm/worktrees/cs-46-parakeet-eou-research-20261020/scripts/speech
export OMP_NUM_THREADS=1 PYTHONPATH=$S
cd $S
CM=$1
echo "[job8/$CM] start $(date -u +%FT%TZ)"
$D/venv-arm/bin/python -I parakeet_eou/parity.py --clips $D/clips.jsonl --nemo-root $D/nemo_ref --nemo-suffix _eval --model-dir $D/dl --export-dir $D/export --out $D/out/parity3_eval_c$CM --chunks $CM --pairings fp32_fp32,fp32_int8dec,fp32_selfdec,fp32_selfdec_int8 > $D/logs/parity3_c$CM.log 2>&1; echo "[job8/$CM] parity rc=$?"
if [ "$CM" = "160" ]; then
  $D/venv-arm/bin/python -I parakeet_eou/compare_parakeet_rs.py --rust-out $D/rust_out --nemo-dir $D/nemo_ref/c160_full_eval --clips $D/clips.jsonl --out $D/out/parakeet_rs_vs_nemo_eval.json > $D/logs/cmp_rs_eval.log 2>&1; echo "[job8] cmp rs eval rc=$?"
  $D/venv-arm/bin/python -I parakeet_eou/compare_parakeet_rs.py --rust-out $D/rust_out --nemo-dir $D/nemo_ref/c160_full --clips $D/clips.jsonl --out $D/out/parakeet_rs_vs_nemo_shipped.json > $D/logs/cmp_rs_shipped.log 2>&1; echo "[job8] cmp rs shipped rc=$?"
  mkdir -p $D/out/events
  for r in none decoder full; do
    $D/venv-arm/bin/python -I parakeet_eou/run_clip.py --wav $D/clips/tt1-p0001-a.wav --model-dir $D/dl --export-dir $D/export --chunk-ms 160 --reset $r --pairing fp32_selfdec --hold-s 1.0 > $D/out/events/tt1-p0001-a_c160_reset_$r.jsonl 2> $D/logs/runclip_$r.err; echo "[job8] runclip $r rc=$?"
  done
  $D/venv-arm/bin/python -I parakeet_eou/run_clip.py --wav $D/clips/tt1-p0001-a.wav --model-dir $D/dl --export-dir $D/export --chunk-ms 80 --reset decoder --pairing fp32_fp32 > $D/out/events/tt1-p0001-a_c80_reset_decoder_comm.jsonl 2> $D/logs/runclip_80.err; echo "[job8] runclip80 rc=$?"
fi
echo "[job8/$CM] done $(date -u +%FT%TZ)"
