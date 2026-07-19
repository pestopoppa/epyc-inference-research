# Qwen3.5-122B IQ2 CPU Prefill Observation

Artifact: `/mnt/raid0/llm/epyc-inference-research/data/cpu_prefill_compute/20260719T014801Z_qwen35_122b_iq2_cpu_prefill`

Command ran from `/mnt/raid0/llm/llama.cpp-experimental`:

```bash
OMP_NUM_THREADS=96 GGML_IQK=1 timeout --preserve-status 2700 /usr/bin/time -v \
  ./build-k24-cpu/bin/llama-bench \
  -m /mnt/raid0/llm/models/Qwen3.5-122B-A10B-MTP-GGUF/UD-IQ2_M/Qwen3.5-122B-A10B-UD-IQ2_M.gguf \
  -p 2048,8192 -n 16 -r 1 --no-warmup -t 96 -b 2048 -ub 512 \
  -ngl 0 --device none -o jsonl --progress
```

Results:

| Row | Tokens/s |
|---|---:|
| `pp2048` | `122.310550` |
| `pp8192` | `114.401943` |
| `tg16` | `6.236179` |

The benchmark reported backend `CPU`, device `none`, GPU info empty,
`n_gpu_layers=0`, and `[iqk] ACTIVE`. `/usr/bin/time` reported `1:33.18`
wall time, max RSS `40062336 KB`, and exit status `0`.

Pre/post `rocm-smi --showpids` both reported no KFD PIDs. Final exact process
checks found no `llama-bench` or `timeout` process.

Interpretation: observation-only. CPU prefill is about `114-122 tok/s` at
2K-8K prompt for the 122B IQ2 candidate, while decode is about `6.24 tok/s`.
This is useful for CPU prefill-compute and hybrid-placement economics; it does
not make CPU-only IQ2 a primary serving lane.
