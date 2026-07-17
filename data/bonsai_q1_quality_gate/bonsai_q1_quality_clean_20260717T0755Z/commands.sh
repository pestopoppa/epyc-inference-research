#!/usr/bin/env bash
set -euo pipefail

export LD_LIBRARY_PATH="/mnt/raid0/llm/llama.cpp-experimental/build-hip/bin"
# experimental binary: /mnt/raid0/llm/llama.cpp-experimental/build-hip/bin/llama-cli

# gate: bonsai_q1_role_claim_gate
# arm: bonsai_q1_cpu_exact_ok
env -i PATH=/usr/bin:/bin LD_LIBRARY_PATH=/mnt/raid0/llm/llama.cpp-experimental/build-hip/bin OMP_NUM_THREADS=1 numactl --interleave=all /mnt/raid0/llm/llama.cpp-experimental/build-hip/bin/llama-cli --device none --device-draft none --simple-io --no-warmup --single-turn --no-display-prompt --no-show-timings --color off -no-cnv --reasoning off --reasoning-budget 0 --temp 0 --seed 1 -m /mnt/raid0/llm/models/bonsai-27b/Bonsai-27B-Q1_0.gguf -t 96 -c 2048 -ngl 0 -n 64 -p 'Return exactly: ok'

# arm: bonsai_q1_mi210_exact_ok
env -i PATH=/usr/bin:/bin LD_LIBRARY_PATH=/mnt/raid0/llm/llama.cpp-experimental/build-hip/bin OMP_NUM_THREADS=1 numactl --interleave=all /mnt/raid0/llm/llama.cpp-experimental/build-hip/bin/llama-cli --device ROCm0 --simple-io --no-warmup --single-turn --no-display-prompt --no-show-timings --color off -no-cnv --reasoning off --reasoning-budget 0 --temp 0 --seed 1 -m /mnt/raid0/llm/models/bonsai-27b/Bonsai-27B-Q1_0.gguf -t 96 -c 2048 -ngl 99 -n 64 -p 'Return exactly: ok'

# arm: bonsai_q1_cpu_strict_json
env -i PATH=/usr/bin:/bin LD_LIBRARY_PATH=/mnt/raid0/llm/llama.cpp-experimental/build-hip/bin OMP_NUM_THREADS=1 numactl --interleave=all /mnt/raid0/llm/llama.cpp-experimental/build-hip/bin/llama-cli --device none --device-draft none --simple-io --no-warmup --single-turn --no-display-prompt --no-show-timings --color off -no-cnv --reasoning off --reasoning-budget 0 --temp 0 --seed 1 -m /mnt/raid0/llm/models/bonsai-27b/Bonsai-27B-Q1_0.gguf -t 96 -c 2048 -ngl 0 -n 96 -p 'Return exactly this minified JSON and nothing else: {"status":"ok","model":"bonsai"}'

# arm: bonsai_q1_mi210_strict_json
env -i PATH=/usr/bin:/bin LD_LIBRARY_PATH=/mnt/raid0/llm/llama.cpp-experimental/build-hip/bin OMP_NUM_THREADS=1 numactl --interleave=all /mnt/raid0/llm/llama.cpp-experimental/build-hip/bin/llama-cli --device ROCm0 --simple-io --no-warmup --single-turn --no-display-prompt --no-show-timings --color off -no-cnv --reasoning off --reasoning-budget 0 --temp 0 --seed 1 -m /mnt/raid0/llm/models/bonsai-27b/Bonsai-27B-Q1_0.gguf -t 96 -c 2048 -ngl 99 -n 96 -p 'Return exactly this minified JSON and nothing else: {"status":"ok","model":"bonsai"}'

# arm: bonsai_q1_cpu_simple_math
env -i PATH=/usr/bin:/bin LD_LIBRARY_PATH=/mnt/raid0/llm/llama.cpp-experimental/build-hip/bin OMP_NUM_THREADS=1 numactl --interleave=all /mnt/raid0/llm/llama.cpp-experimental/build-hip/bin/llama-cli --device none --device-draft none --simple-io --no-warmup --single-turn --no-display-prompt --no-show-timings --color off -no-cnv --reasoning off --reasoning-budget 0 --temp 0 --seed 1 -m /mnt/raid0/llm/models/bonsai-27b/Bonsai-27B-Q1_0.gguf -t 96 -c 2048 -ngl 0 -n 32 -p 'Answer with only the integer result: 37 + 58'

# arm: bonsai_q1_mi210_simple_math
env -i PATH=/usr/bin:/bin LD_LIBRARY_PATH=/mnt/raid0/llm/llama.cpp-experimental/build-hip/bin OMP_NUM_THREADS=1 numactl --interleave=all /mnt/raid0/llm/llama.cpp-experimental/build-hip/bin/llama-cli --device ROCm0 --simple-io --no-warmup --single-turn --no-display-prompt --no-show-timings --color off -no-cnv --reasoning off --reasoning-budget 0 --temp 0 --seed 1 -m /mnt/raid0/llm/models/bonsai-27b/Bonsai-27B-Q1_0.gguf -t 96 -c 2048 -ngl 99 -n 32 -p 'Answer with only the integer result: 37 + 58'

# arm: bonsai_q1_cpu_short_instruction
env -i PATH=/usr/bin:/bin LD_LIBRARY_PATH=/mnt/raid0/llm/llama.cpp-experimental/build-hip/bin OMP_NUM_THREADS=1 numactl --interleave=all /mnt/raid0/llm/llama.cpp-experimental/build-hip/bin/llama-cli --device none --device-draft none --simple-io --no-warmup --single-turn --no-display-prompt --no-show-timings --color off -no-cnv --reasoning off --reasoning-budget 0 --temp 0 --seed 1 -m /mnt/raid0/llm/models/bonsai-27b/Bonsai-27B-Q1_0.gguf -t 96 -c 2048 -ngl 0 -n 48 -p 'In exactly six lowercase words, describe why benchmarks need held-out tests.'

# arm: bonsai_q1_mi210_short_instruction
env -i PATH=/usr/bin:/bin LD_LIBRARY_PATH=/mnt/raid0/llm/llama.cpp-experimental/build-hip/bin OMP_NUM_THREADS=1 numactl --interleave=all /mnt/raid0/llm/llama.cpp-experimental/build-hip/bin/llama-cli --device ROCm0 --simple-io --no-warmup --single-turn --no-display-prompt --no-show-timings --color off -no-cnv --reasoning off --reasoning-budget 0 --temp 0 --seed 1 -m /mnt/raid0/llm/models/bonsai-27b/Bonsai-27B-Q1_0.gguf -t 96 -c 2048 -ngl 99 -n 48 -p 'In exactly six lowercase words, describe why benchmarks need held-out tests.'
