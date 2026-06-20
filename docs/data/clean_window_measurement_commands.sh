#!/usr/bin/env bash
set -euo pipefail
# Generated at 2026-06-20T20:21:12.309355+00:00
# Review live topology before running commands with direct --port values.

# model_path: /mnt/raid0/llm/lmstudio/models/lmstudio-community/Qwen3-Next-80B-A3B-Instruct-GGUF/Qwen3-Next-80B-A3B-Instruct-Q4_K_M.gguf
# roles: ingest_long_context
# K-MEM-1 ingest_long_context run_benchmark_suite tulving_episodic [ready]
/mnt/raid0/llm/epyc-inference-research/.venv/bin/python3 /mnt/raid0/llm/epyc-inference-research/scripts/benchmark/run_benchmark.py --model ingest_long_context --suite tulving_episodic --new-run --server-mode --skip-speed-tests

# K-ROPE-1 ingest_long_context rope_position_probe ctx=4096 [ready]
/mnt/raid0/llm/epyc-inference-research/.venv/bin/python3 /mnt/raid0/llm/epyc-inference-research/scripts/benchmark/rope_position_probe.py --api chat --host 127.0.0.1 --port 8085 --context-length 4096 --n-samples 100 --seed 42 --out /mnt/raid0/llm/epyc-inference-research/benchmarks/results/clean_window/rope_probe/ingest_long_context/ctx_4096.json

# K-ROPE-1 ingest_long_context rope_position_probe ctx=8192 [ready]
/mnt/raid0/llm/epyc-inference-research/.venv/bin/python3 /mnt/raid0/llm/epyc-inference-research/scripts/benchmark/rope_position_probe.py --api chat --host 127.0.0.1 --port 8085 --context-length 8192 --n-samples 100 --seed 42 --out /mnt/raid0/llm/epyc-inference-research/benchmarks/results/clean_window/rope_probe/ingest_long_context/ctx_8192.json

# K-ROPE-1 ingest_long_context rope_position_probe ctx=16384 [ready]
/mnt/raid0/llm/epyc-inference-research/.venv/bin/python3 /mnt/raid0/llm/epyc-inference-research/scripts/benchmark/rope_position_probe.py --api chat --host 127.0.0.1 --port 8085 --context-length 16384 --n-samples 100 --seed 42 --out /mnt/raid0/llm/epyc-inference-research/benchmarks/results/clean_window/rope_probe/ingest_long_context/ctx_16384.json

# K-ROPE-1 ingest_long_context rope_position_probe ctx=32768 [blocked]
# blocked: /mnt/raid0/llm/epyc-inference-research/.venv/bin/python3 /mnt/raid0/llm/epyc-inference-research/scripts/benchmark/rope_position_probe.py --api chat --host 127.0.0.1 --port 8085 --context-length 32768 --n-samples 100 --seed 42 --out /mnt/raid0/llm/epyc-inference-research/benchmarks/results/clean_window/rope_probe/ingest_long_context/ctx_32768.json
# note: context 32768 needs chat-template headroom below live server context 32768

# model_path: /mnt/raid0/llm/lmstudio/models/unsloth/Qwen3.5-122B-A10B-GGUF/Q4_K_M/Qwen3.5-122B-A10B-Q4_K_M-00001-of-00003.gguf
# roles: architect_general
# G10 architect_general run_benchmark_suite omniscience [ready]
/mnt/raid0/llm/epyc-inference-research/.venv/bin/python3 /mnt/raid0/llm/epyc-inference-research/scripts/benchmark/run_benchmark.py --model architect_general --suite omniscience --new-run --server-mode --skip-speed-tests

# K-ROPE-1 architect_general rope_position_probe ctx=4096 [ready]
/mnt/raid0/llm/epyc-inference-research/.venv/bin/python3 /mnt/raid0/llm/epyc-inference-research/scripts/benchmark/rope_position_probe.py --api chat --host 127.0.0.1 --port 8083 --context-length 4096 --n-samples 100 --seed 42 --out /mnt/raid0/llm/epyc-inference-research/benchmarks/results/clean_window/rope_probe/architect_general/ctx_4096.json

# K-ROPE-1 architect_general rope_position_probe ctx=8192 [ready]
/mnt/raid0/llm/epyc-inference-research/.venv/bin/python3 /mnt/raid0/llm/epyc-inference-research/scripts/benchmark/rope_position_probe.py --api chat --host 127.0.0.1 --port 8083 --context-length 8192 --n-samples 100 --seed 42 --out /mnt/raid0/llm/epyc-inference-research/benchmarks/results/clean_window/rope_probe/architect_general/ctx_8192.json

# K-ROPE-1 architect_general rope_position_probe ctx=16384 [blocked]
# blocked: /mnt/raid0/llm/epyc-inference-research/.venv/bin/python3 /mnt/raid0/llm/epyc-inference-research/scripts/benchmark/rope_position_probe.py --api chat --host 127.0.0.1 --port 8083 --context-length 16384 --n-samples 100 --seed 42 --out /mnt/raid0/llm/epyc-inference-research/benchmarks/results/clean_window/rope_probe/architect_general/ctx_16384.json
# note: context 16384 needs chat-template headroom below live server context 16384

# K-ROPE-1 architect_general rope_position_probe ctx=32768 [blocked]
# blocked: /mnt/raid0/llm/epyc-inference-research/.venv/bin/python3 /mnt/raid0/llm/epyc-inference-research/scripts/benchmark/rope_position_probe.py --api chat --host 127.0.0.1 --port 8083 --context-length 32768 --n-samples 100 --seed 42 --out /mnt/raid0/llm/epyc-inference-research/benchmarks/results/clean_window/rope_probe/architect_general/ctx_32768.json
# note: context 32768 needs chat-template headroom below live server context 16384

# G5 architect_general short_mk_voting [ready]
/mnt/raid0/llm/epyc-inference-research/.venv/bin/python3 /mnt/raid0/llm/epyc-inference-research/scripts/benchmark/short_mk_voting.py --role architect_general --host 127.0.0.1 --model-port 8083 --suites gpqa math --sample-per-suite 20 --k 3 --m 3 --sequential --output /mnt/raid0/llm/epyc-inference-research/benchmarks/results/clean_window/short_mk_voting/architect_general.json

# model_path: /mnt/raid0/llm/models/Qwen_Qwen3.6-35B-A3B-Q8_0.gguf
# roles: frontdoor
# G11 frontdoor run_benchmark_suite omniscience [ready]
/mnt/raid0/llm/epyc-inference-research/.venv/bin/python3 /mnt/raid0/llm/epyc-inference-research/scripts/benchmark/run_benchmark.py --model frontdoor --suite omniscience --new-run --server-mode --skip-speed-tests

# K-ROPE-1 frontdoor rope_position_probe ctx=4096 [ready]
/mnt/raid0/llm/epyc-inference-research/.venv/bin/python3 /mnt/raid0/llm/epyc-inference-research/scripts/benchmark/rope_position_probe.py --api chat --host 127.0.0.1 --port 8070 --context-length 4096 --n-samples 100 --seed 42 --out /mnt/raid0/llm/epyc-inference-research/benchmarks/results/clean_window/rope_probe/frontdoor/ctx_4096.json

# K-ROPE-1 frontdoor rope_position_probe ctx=8192 [ready]
/mnt/raid0/llm/epyc-inference-research/.venv/bin/python3 /mnt/raid0/llm/epyc-inference-research/scripts/benchmark/rope_position_probe.py --api chat --host 127.0.0.1 --port 8070 --context-length 8192 --n-samples 100 --seed 42 --out /mnt/raid0/llm/epyc-inference-research/benchmarks/results/clean_window/rope_probe/frontdoor/ctx_8192.json

# K-ROPE-1 frontdoor rope_position_probe ctx=16384 [ready]
/mnt/raid0/llm/epyc-inference-research/.venv/bin/python3 /mnt/raid0/llm/epyc-inference-research/scripts/benchmark/rope_position_probe.py --api chat --host 127.0.0.1 --port 8070 --context-length 16384 --n-samples 100 --seed 42 --out /mnt/raid0/llm/epyc-inference-research/benchmarks/results/clean_window/rope_probe/frontdoor/ctx_16384.json

# K-ROPE-1 frontdoor rope_position_probe ctx=32768 [blocked]
# blocked: /mnt/raid0/llm/epyc-inference-research/.venv/bin/python3 /mnt/raid0/llm/epyc-inference-research/scripts/benchmark/rope_position_probe.py --api chat --host 127.0.0.1 --port 8070 --context-length 32768 --n-samples 100 --seed 42 --out /mnt/raid0/llm/epyc-inference-research/benchmarks/results/clean_window/rope_probe/frontdoor/ctx_32768.json
# note: context 32768 needs chat-template headroom below live server context 32768

# G5 frontdoor short_mk_voting [ready]
/mnt/raid0/llm/epyc-inference-research/.venv/bin/python3 /mnt/raid0/llm/epyc-inference-research/scripts/benchmark/short_mk_voting.py --role frontdoor --host 127.0.0.1 --model-port 8070 --suites gpqa math --sample-per-suite 20 --k 3 --m 3 --sequential --output /mnt/raid0/llm/epyc-inference-research/benchmarks/results/clean_window/short_mk_voting/frontdoor.json

# model_path: /mnt/raid0/llm/models/gemma-4-26B-A4B-it-Q4_K_M.gguf
# roles: worker_general
# G11 worker_general run_benchmark_suite omniscience [ready]
/mnt/raid0/llm/epyc-inference-research/.venv/bin/python3 /mnt/raid0/llm/epyc-inference-research/scripts/benchmark/run_benchmark.py --model worker_general --suite omniscience --new-run --server-mode --skip-speed-tests

# K-ROPE-1 worker_general rope_position_probe ctx=4096 [ready]
/mnt/raid0/llm/epyc-inference-research/.venv/bin/python3 /mnt/raid0/llm/epyc-inference-research/scripts/benchmark/rope_position_probe.py --api chat --host 127.0.0.1 --port 8072 --context-length 4096 --n-samples 100 --seed 42 --out /mnt/raid0/llm/epyc-inference-research/benchmarks/results/clean_window/rope_probe/worker_general/ctx_4096.json

# K-ROPE-1 worker_general rope_position_probe ctx=8192 [ready]
/mnt/raid0/llm/epyc-inference-research/.venv/bin/python3 /mnt/raid0/llm/epyc-inference-research/scripts/benchmark/rope_position_probe.py --api chat --host 127.0.0.1 --port 8072 --context-length 8192 --n-samples 100 --seed 42 --out /mnt/raid0/llm/epyc-inference-research/benchmarks/results/clean_window/rope_probe/worker_general/ctx_8192.json

# K-ROPE-1 worker_general rope_position_probe ctx=16384 [blocked]
# blocked: /mnt/raid0/llm/epyc-inference-research/.venv/bin/python3 /mnt/raid0/llm/epyc-inference-research/scripts/benchmark/rope_position_probe.py --api chat --host 127.0.0.1 --port 8072 --context-length 16384 --n-samples 100 --seed 42 --out /mnt/raid0/llm/epyc-inference-research/benchmarks/results/clean_window/rope_probe/worker_general/ctx_16384.json
# note: context 16384 needs chat-template headroom below live server context 16384

# K-ROPE-1 worker_general rope_position_probe ctx=32768 [blocked]
# blocked: /mnt/raid0/llm/epyc-inference-research/.venv/bin/python3 /mnt/raid0/llm/epyc-inference-research/scripts/benchmark/rope_position_probe.py --api chat --host 127.0.0.1 --port 8072 --context-length 32768 --n-samples 100 --seed 42 --out /mnt/raid0/llm/epyc-inference-research/benchmarks/results/clean_window/rope_probe/worker_general/ctx_32768.json
# note: context 32768 needs chat-template headroom below live server context 16384

# G5 worker_general short_mk_voting [ready]
/mnt/raid0/llm/epyc-inference-research/.venv/bin/python3 /mnt/raid0/llm/epyc-inference-research/scripts/benchmark/short_mk_voting.py --role worker_general --host 127.0.0.1 --model-port 8072 --suites gpqa math --sample-per-suite 20 --k 3 --m 3 --sequential --output /mnt/raid0/llm/epyc-inference-research/benchmarks/results/clean_window/short_mk_voting/worker_general.json

# model_path: clean-window-harness:ds-e1-kv
# roles: dynamic_stack
# DS-E1 dynamic_stack production_kv_measurements [ready]
bash /mnt/raid0/llm/epyc-inference-research/scripts/benchmark/ds_e1_kv_measurements.sh --execute

# model_path: clean-window-harness:xmas-constrained-policy
# roles: xmas_routing
# X-MAS xmas_routing constrained_policy_heldout_ab [ready]
cd /mnt/raid0/llm/epyc-orchestrator && uv run python scripts/benchmark/xmas_live_ab.py --prompts benchmarks/results/runs/xmas_live_ab/20260618-heldout-resilient/prompts.jsonl --reps 2 --host-quiet-confirmed --output benchmarks/results/runs/xmas_live_ab/$(date -u +%Y%m%dT%H%M%SZ)-constrained-policy
