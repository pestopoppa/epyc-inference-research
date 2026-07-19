# Nemotron-Nano 9B Q8 CPU source-head v7 context rerun

- Artifact: `/mnt/raid0/llm/epyc-inference-research/data/nemotron/nemotron-nano-9b-q8-cpu-sourcehead-v7-context-20260718T235237Z`
- Exit code: `0`
- Experimental commit: `ed4091266d286045510e498ceb059c209a65aff9`
- Binary mtime: `2026-07-18 22:02:00.309135337 +0000`
- Cleanup: `pid_not_present_after_wait`; final residual check: `none`; signals: `none`

## Command

```bash
LD_LIBRARY_PATH=/mnt/raid0/llm/llama.cpp-experimental/build-hip/bin \
/mnt/raid0/llm/llama.cpp-experimental/build-hip/bin/llama-bench \
  -m /mnt/raid0/llm/models/Nemotron-Nano-9B-v2-GGUF/nvidia_NVIDIA-Nemotron-Nano-9B-v2-Q8_0.gguf \
  -dev none -ngl 0 -fa off -t 16 -n 512 \
  -pg 512,0 -pg 0,512 -pg 2048,512 -pg 8192,512 -pg 32768,512 \
  -r 1 -o json --progress
```

## t/s rows

| row | test | n_prompt | n_gen | avg t/s | stddev t/s | samples t/s |
|---:|---|---:|---:|---:|---:|---|
| 1 | pp512 | 512 | 0 | 133.214857 | 0.000000 | 133.215 |
| 2 | tg512 | 0 | 512 | 5.482631 | 0.000000 | 5.48263 |
| 3 | pp512 | 512 | 0 | 138.037227 | 0.000000 | 138.037 |
| 4 | tg512 | 0 | 512 | 5.498687 | 0.000000 | 5.49869 |
| 5 | pp2048+tg512 | 2048 | 512 | 23.533154 | 0.000000 | 23.5332 |
| 6 | pp8192+tg512 | 8192 | 512 | 53.305435 | 0.000000 | 53.3054 |
| 7 | pp32768+tg512 | 32768 | 512 | 76.864739 | 0.000000 | 76.8647 |

## Caveats

- HIP build stderr reports ROCm device discovery even though the command used `-dev none -ngl 0`; raw pre/post `rocm-smi` snapshots are included.
- Pre-run `rocm-smi` showed unrelated GPU activity already present (`VRAM% 40`, `GPU% 99`); final post-run snapshot showed `VRAM% 0`, `GPU% 0`.
- `llama-bench` produced 7 rows: the requested `-n 512` default prompt/generation rows plus the five requested `-pg` rows.
- Final `post_pgrep.txt` still includes unrelated long-lived processes whose command lines contain `llama`; the exact benchmark PID and exact experimental benchmark command are absent.
