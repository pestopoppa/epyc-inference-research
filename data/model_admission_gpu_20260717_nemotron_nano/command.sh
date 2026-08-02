export LD_LIBRARY_PATH="/mnt/raid0/llm/llama.cpp-experimental/build-hip/bin"
export HIP_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1
timeout --signal=KILL 1800 "/mnt/raid0/llm/llama.cpp-experimental/build-hip/bin/llama-cli" --device ROCm0 --simple-io --no-warmup --single-turn --no-display-prompt --color off -m "/mnt/raid0/llm/models/Nemotron-Nano-9B-v2-GGUF/nvidia_NVIDIA-Nemotron-Nano-9B-v2-Q8_0.gguf" -ngl 99 -c 2048 -n 1536 --ignore-eos --reasoning-format none --reasoning off --temp 0.8 --top-k 40 --top-p 0.95 -p @"/mnt/raid0/llm/tmp/model-admission-gpu-20260717-nemotron-nano/prompt.txt"
