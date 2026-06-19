#!/usr/bin/env bash
set -euo pipefail
# Generated at 2026-06-19T00:55:05.551902+00:00
# Review live topology before running commands with direct --port values.

# model_path: /mnt/raid0/llm/lmstudio/models/lmstudio-community/Qwen3-Next-80B-A3B-Instruct-GGUF/Qwen3-Next-80B-A3B-Instruct-Q4_K_M.gguf
# roles: ingest_long_context
# K-MEM-1 ingest_long_context run_benchmark_suite tulving_episodic [ready]
/mnt/raid0/llm/epyc-inference-research/.venv/bin/python3 /mnt/raid0/llm/epyc-inference-research/scripts/benchmark/run_benchmark.py --model ingest_long_context --suite tulving_episodic --new-run --server-mode

# K-ROPE-1 ingest_long_context rope_position_probe ctx=4096 [ready]
/mnt/raid0/llm/epyc-inference-research/.venv/bin/python3 /mnt/raid0/llm/epyc-inference-research/scripts/benchmark/rope_position_probe.py --host 127.0.0.1 --port 8085 --context-length 4096 --n-samples 100 --seed 42 --out /mnt/raid0/llm/epyc-inference-research/benchmarks/results/clean_window/rope_probe/ingest_long_context/ctx_4096.json

# K-ROPE-1 ingest_long_context rope_position_probe ctx=8192 [ready]
/mnt/raid0/llm/epyc-inference-research/.venv/bin/python3 /mnt/raid0/llm/epyc-inference-research/scripts/benchmark/rope_position_probe.py --host 127.0.0.1 --port 8085 --context-length 8192 --n-samples 100 --seed 42 --out /mnt/raid0/llm/epyc-inference-research/benchmarks/results/clean_window/rope_probe/ingest_long_context/ctx_8192.json

# K-ROPE-1 ingest_long_context rope_position_probe ctx=16384 [ready]
/mnt/raid0/llm/epyc-inference-research/.venv/bin/python3 /mnt/raid0/llm/epyc-inference-research/scripts/benchmark/rope_position_probe.py --host 127.0.0.1 --port 8085 --context-length 16384 --n-samples 100 --seed 42 --out /mnt/raid0/llm/epyc-inference-research/benchmarks/results/clean_window/rope_probe/ingest_long_context/ctx_16384.json

# K-ROPE-1 ingest_long_context rope_position_probe ctx=32768 [ready]
/mnt/raid0/llm/epyc-inference-research/.venv/bin/python3 /mnt/raid0/llm/epyc-inference-research/scripts/benchmark/rope_position_probe.py --host 127.0.0.1 --port 8085 --context-length 32768 --n-samples 100 --seed 42 --out /mnt/raid0/llm/epyc-inference-research/benchmarks/results/clean_window/rope_probe/ingest_long_context/ctx_32768.json

# model_path: /mnt/raid0/llm/lmstudio/models/unsloth/Qwen3.5-122B-A10B-GGUF/Q4_K_M/Qwen3.5-122B-A10B-Q4_K_M-00001-of-00003.gguf
# roles: architect_general
# G10 architect_general run_benchmark_suite omniscience [ready]
/mnt/raid0/llm/epyc-inference-research/.venv/bin/python3 /mnt/raid0/llm/epyc-inference-research/scripts/benchmark/run_benchmark.py --model architect_general --suite omniscience --new-run

# K-ROPE-1 architect_general rope_position_probe ctx=4096 [ready]
/mnt/raid0/llm/epyc-inference-research/.venv/bin/python3 /mnt/raid0/llm/epyc-inference-research/scripts/benchmark/rope_position_probe.py --host 127.0.0.1 --port 8083 --context-length 4096 --n-samples 100 --seed 42 --out /mnt/raid0/llm/epyc-inference-research/benchmarks/results/clean_window/rope_probe/architect_general/ctx_4096.json

# K-ROPE-1 architect_general rope_position_probe ctx=8192 [ready]
/mnt/raid0/llm/epyc-inference-research/.venv/bin/python3 /mnt/raid0/llm/epyc-inference-research/scripts/benchmark/rope_position_probe.py --host 127.0.0.1 --port 8083 --context-length 8192 --n-samples 100 --seed 42 --out /mnt/raid0/llm/epyc-inference-research/benchmarks/results/clean_window/rope_probe/architect_general/ctx_8192.json

# K-ROPE-1 architect_general rope_position_probe ctx=16384 [ready]
/mnt/raid0/llm/epyc-inference-research/.venv/bin/python3 /mnt/raid0/llm/epyc-inference-research/scripts/benchmark/rope_position_probe.py --host 127.0.0.1 --port 8083 --context-length 16384 --n-samples 100 --seed 42 --out /mnt/raid0/llm/epyc-inference-research/benchmarks/results/clean_window/rope_probe/architect_general/ctx_16384.json

# K-ROPE-1 architect_general rope_position_probe ctx=32768 [blocked]
# blocked: /mnt/raid0/llm/epyc-inference-research/.venv/bin/python3 /mnt/raid0/llm/epyc-inference-research/scripts/benchmark/rope_position_probe.py --host 127.0.0.1 --port 8083 --context-length 32768 --n-samples 100 --seed 42 --out /mnt/raid0/llm/epyc-inference-research/benchmarks/results/clean_window/rope_probe/architect_general/ctx_32768.json
# note: context 32768 exceeds live server context 16384

# G5 architect_general short_mk_voting [blocked]
# note: no short-m@k voting runner found; G5 needs runner wiring before clean-window execution

# model_path: /mnt/raid0/llm/models/Qwen_Qwen3.6-35B-A3B-Q8_0.gguf
# roles: frontdoor
# G11 frontdoor run_benchmark_suite omniscience [ready]
/mnt/raid0/llm/epyc-inference-research/.venv/bin/python3 /mnt/raid0/llm/epyc-inference-research/scripts/benchmark/run_benchmark.py --model frontdoor --suite omniscience --new-run

# K-ROPE-1 frontdoor rope_position_probe ctx=4096 [ready]
/mnt/raid0/llm/epyc-inference-research/.venv/bin/python3 /mnt/raid0/llm/epyc-inference-research/scripts/benchmark/rope_position_probe.py --host 127.0.0.1 --port 8070 --context-length 4096 --n-samples 100 --seed 42 --out /mnt/raid0/llm/epyc-inference-research/benchmarks/results/clean_window/rope_probe/frontdoor/ctx_4096.json

# K-ROPE-1 frontdoor rope_position_probe ctx=8192 [ready]
/mnt/raid0/llm/epyc-inference-research/.venv/bin/python3 /mnt/raid0/llm/epyc-inference-research/scripts/benchmark/rope_position_probe.py --host 127.0.0.1 --port 8070 --context-length 8192 --n-samples 100 --seed 42 --out /mnt/raid0/llm/epyc-inference-research/benchmarks/results/clean_window/rope_probe/frontdoor/ctx_8192.json

# K-ROPE-1 frontdoor rope_position_probe ctx=16384 [ready]
/mnt/raid0/llm/epyc-inference-research/.venv/bin/python3 /mnt/raid0/llm/epyc-inference-research/scripts/benchmark/rope_position_probe.py --host 127.0.0.1 --port 8070 --context-length 16384 --n-samples 100 --seed 42 --out /mnt/raid0/llm/epyc-inference-research/benchmarks/results/clean_window/rope_probe/frontdoor/ctx_16384.json

# K-ROPE-1 frontdoor rope_position_probe ctx=32768 [ready]
/mnt/raid0/llm/epyc-inference-research/.venv/bin/python3 /mnt/raid0/llm/epyc-inference-research/scripts/benchmark/rope_position_probe.py --host 127.0.0.1 --port 8070 --context-length 32768 --n-samples 100 --seed 42 --out /mnt/raid0/llm/epyc-inference-research/benchmarks/results/clean_window/rope_probe/frontdoor/ctx_32768.json

# G5 frontdoor short_mk_voting [blocked]
# note: no short-m@k voting runner found; G5 needs runner wiring before clean-window execution

# model_path: /mnt/raid0/llm/models/gemma-4-26B-A4B-it-Q4_K_M.gguf
# roles: worker_general
# G11 worker_general run_benchmark_suite omniscience [blocked]
# blocked: /mnt/raid0/llm/epyc-inference-research/.venv/bin/python3 /mnt/raid0/llm/epyc-inference-research/scripts/benchmark/run_benchmark.py --model worker_general --suite omniscience --new-run
# note: benchmark registry model path differs from live registry; run_benchmark.py would not measure the live role

# K-ROPE-1 worker_general rope_position_probe ctx=4096 [ready]
/mnt/raid0/llm/epyc-inference-research/.venv/bin/python3 /mnt/raid0/llm/epyc-inference-research/scripts/benchmark/rope_position_probe.py --host 127.0.0.1 --port 8072 --context-length 4096 --n-samples 100 --seed 42 --out /mnt/raid0/llm/epyc-inference-research/benchmarks/results/clean_window/rope_probe/worker_general/ctx_4096.json

# K-ROPE-1 worker_general rope_position_probe ctx=8192 [ready]
/mnt/raid0/llm/epyc-inference-research/.venv/bin/python3 /mnt/raid0/llm/epyc-inference-research/scripts/benchmark/rope_position_probe.py --host 127.0.0.1 --port 8072 --context-length 8192 --n-samples 100 --seed 42 --out /mnt/raid0/llm/epyc-inference-research/benchmarks/results/clean_window/rope_probe/worker_general/ctx_8192.json

# K-ROPE-1 worker_general rope_position_probe ctx=16384 [ready]
/mnt/raid0/llm/epyc-inference-research/.venv/bin/python3 /mnt/raid0/llm/epyc-inference-research/scripts/benchmark/rope_position_probe.py --host 127.0.0.1 --port 8072 --context-length 16384 --n-samples 100 --seed 42 --out /mnt/raid0/llm/epyc-inference-research/benchmarks/results/clean_window/rope_probe/worker_general/ctx_16384.json

# K-ROPE-1 worker_general rope_position_probe ctx=32768 [blocked]
# blocked: /mnt/raid0/llm/epyc-inference-research/.venv/bin/python3 /mnt/raid0/llm/epyc-inference-research/scripts/benchmark/rope_position_probe.py --host 127.0.0.1 --port 8072 --context-length 32768 --n-samples 100 --seed 42 --out /mnt/raid0/llm/epyc-inference-research/benchmarks/results/clean_window/rope_probe/worker_general/ctx_32768.json
# note: context 32768 exceeds live server context 16384

# G5 worker_general short_mk_voting [blocked]
# note: no short-m@k voting runner found; G5 needs runner wiring before clean-window execution
