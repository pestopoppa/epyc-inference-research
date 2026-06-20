#!/usr/bin/env bash
set -euo pipefail
# Generated at 2026-06-20T04:29:32.238533+00:00
# Suite: omniscience

# architect_general [ready]
/mnt/raid0/llm/epyc-inference-research/.venv/bin/python3 /mnt/raid0/llm/epyc-inference-research/scripts/benchmark/run_benchmark.py --model architect_general --suite omniscience --new-run --server-mode --skip-speed-tests

# architect_hermes_4_70b [blocked]
# blocked: /mnt/raid0/llm/epyc-inference-research/.venv/bin/python3 /mnt/raid0/llm/epyc-inference-research/scripts/benchmark/run_benchmark.py --model architect_hermes_4_70b --suite omniscience --new-run --server-mode --skip-speed-tests

# architect_qwen2_5_72b [blocked]
# blocked: /mnt/raid0/llm/epyc-inference-research/.venv/bin/python3 /mnt/raid0/llm/epyc-inference-research/scripts/benchmark/run_benchmark.py --model architect_qwen2_5_72b --suite omniscience --new-run --server-mode --skip-speed-tests

# architect_qwen2_5_72b_q4_k_m [blocked]
# blocked: /mnt/raid0/llm/epyc-inference-research/.venv/bin/python3 /mnt/raid0/llm/epyc-inference-research/scripts/benchmark/run_benchmark.py --model architect_qwen2_5_72b_q4_k_m --suite omniscience --new-run --server-mode --skip-speed-tests

# coder_escalation [ready]
/mnt/raid0/llm/epyc-inference-research/.venv/bin/python3 /mnt/raid0/llm/epyc-inference-research/scripts/benchmark/run_benchmark.py --model coder_escalation --suite omniscience --new-run --server-mode --skip-speed-tests

# coder_escalation_q8 [ready]
/mnt/raid0/llm/epyc-inference-research/.venv/bin/python3 /mnt/raid0/llm/epyc-inference-research/scripts/benchmark/run_benchmark.py --model coder_escalation_q8 --suite omniscience --new-run --server-mode --skip-speed-tests

# coder_qwen3_coder_30b_a3b [ready]
/mnt/raid0/llm/epyc-inference-research/.venv/bin/python3 /mnt/raid0/llm/epyc-inference-research/scripts/benchmark/run_benchmark.py --model coder_qwen3_coder_30b_a3b --suite omniscience --new-run --server-mode --skip-speed-tests

# external_claude_opus [blocked]
# blocked: /mnt/raid0/llm/epyc-inference-research/.venv/bin/python3 /mnt/raid0/llm/epyc-inference-research/scripts/benchmark/run_benchmark.py --model external_claude_opus --suite omniscience --new-run --server-mode --skip-speed-tests

# external_claude_sonnet [blocked]
# blocked: /mnt/raid0/llm/epyc-inference-research/.venv/bin/python3 /mnt/raid0/llm/epyc-inference-research/scripts/benchmark/run_benchmark.py --model external_claude_sonnet --suite omniscience --new-run --server-mode --skip-speed-tests

# external_gpt4o [blocked]
# blocked: /mnt/raid0/llm/epyc-inference-research/.venv/bin/python3 /mnt/raid0/llm/epyc-inference-research/scripts/benchmark/run_benchmark.py --model external_gpt4o --suite omniscience --new-run --server-mode --skip-speed-tests

# external_gpt4o_mini [blocked]
# blocked: /mnt/raid0/llm/epyc-inference-research/.venv/bin/python3 /mnt/raid0/llm/epyc-inference-research/scripts/benchmark/run_benchmark.py --model external_gpt4o_mini --suite omniscience --new-run --server-mode --skip-speed-tests

# frontdoor [ready]
/mnt/raid0/llm/epyc-inference-research/.venv/bin/python3 /mnt/raid0/llm/epyc-inference-research/scripts/benchmark/run_benchmark.py --model frontdoor --suite omniscience --new-run --server-mode --skip-speed-tests

# gemma4_26b_a4b_q4km_mtp [ready]
/mnt/raid0/llm/epyc-inference-research/.venv/bin/python3 /mnt/raid0/llm/epyc-inference-research/scripts/benchmark/run_benchmark.py --model gemma4_26b_a4b_q4km_mtp --suite omniscience --new-run --server-mode --skip-speed-tests

# gemma4_31b_q4km_mtp [ready]
/mnt/raid0/llm/epyc-inference-research/.venv/bin/python3 /mnt/raid0/llm/epyc-inference-research/scripts/benchmark/run_benchmark.py --model gemma4_31b_q4km_mtp --suite omniscience --new-run --server-mode --skip-speed-tests

# general_deepseek_r1_0528_qwen3_8b [ready]
/mnt/raid0/llm/epyc-inference-research/.venv/bin/python3 /mnt/raid0/llm/epyc-inference-research/scripts/benchmark/run_benchmark.py --model general_deepseek_r1_0528_qwen3_8b --suite omniscience --new-run --server-mode --skip-speed-tests

# general_gemma_3_12b_it [ready]
/mnt/raid0/llm/epyc-inference-research/.venv/bin/python3 /mnt/raid0/llm/epyc-inference-research/scripts/benchmark/run_benchmark.py --model general_gemma_3_12b_it --suite omniscience --new-run --server-mode --skip-speed-tests

# general_gemma_3_27b_it_qat [blocked]
# blocked: /mnt/raid0/llm/epyc-inference-research/.venv/bin/python3 /mnt/raid0/llm/epyc-inference-research/scripts/benchmark/run_benchmark.py --model general_gemma_3_27b_it_qat --suite omniscience --new-run --server-mode --skip-speed-tests

# general_meta_llama_3_1_8b_q4_k_s [ready]
/mnt/raid0/llm/epyc-inference-research/.venv/bin/python3 /mnt/raid0/llm/epyc-inference-research/scripts/benchmark/run_benchmark.py --model general_meta_llama_3_1_8b_q4_k_s --suite omniscience --new-run --server-mode --skip-speed-tests

# general_meta_llama_3_8b_instruct_fp16 [blocked]
# blocked: /mnt/raid0/llm/epyc-inference-research/.venv/bin/python3 /mnt/raid0/llm/epyc-inference-research/scripts/benchmark/run_benchmark.py --model general_meta_llama_3_8b_instruct_fp16 --suite omniscience --new-run --server-mode --skip-speed-tests

# general_qwen2_5_7b_q4_k_s [ready]
/mnt/raid0/llm/epyc-inference-research/.venv/bin/python3 /mnt/raid0/llm/epyc-inference-research/scripts/benchmark/run_benchmark.py --model general_qwen2_5_7b_q4_k_s --suite omniscience --new-run --server-mode --skip-speed-tests

# general_qwen3_32b [ready]
/mnt/raid0/llm/epyc-inference-research/.venv/bin/python3 /mnt/raid0/llm/epyc-inference-research/scripts/benchmark/run_benchmark.py --model general_qwen3_32b --suite omniscience --new-run --server-mode --skip-speed-tests

# glm_47_flash [ready]
/mnt/raid0/llm/epyc-inference-research/.venv/bin/python3 /mnt/raid0/llm/epyc-inference-research/scripts/benchmark/run_benchmark.py --model glm_47_flash --suite omniscience --new-run --server-mode --skip-speed-tests

# ingest_hermes_4_70b [blocked]
# blocked: /mnt/raid0/llm/epyc-inference-research/.venv/bin/python3 /mnt/raid0/llm/epyc-inference-research/scripts/benchmark/run_benchmark.py --model ingest_hermes_4_70b --suite omniscience --new-run --server-mode --skip-speed-tests

# ingest_long_context [ready]
/mnt/raid0/llm/epyc-inference-research/.venv/bin/python3 /mnt/raid0/llm/epyc-inference-research/scripts/benchmark/run_benchmark.py --model ingest_long_context --suite omniscience --new-run --server-mode --skip-speed-tests

# ingest_qwen2_5_72b [blocked]
# blocked: /mnt/raid0/llm/epyc-inference-research/.venv/bin/python3 /mnt/raid0/llm/epyc-inference-research/scripts/benchmark/run_benchmark.py --model ingest_qwen2_5_72b --suite omniscience --new-run --server-mode --skip-speed-tests

# ingest_qwen2_5_coder_32b [ready]
/mnt/raid0/llm/epyc-inference-research/.venv/bin/python3 /mnt/raid0/llm/epyc-inference-research/scripts/benchmark/run_benchmark.py --model ingest_qwen2_5_coder_32b --suite omniscience --new-run --server-mode --skip-speed-tests

# ingest_qwen3_32b [ready]
/mnt/raid0/llm/epyc-inference-research/.venv/bin/python3 /mnt/raid0/llm/epyc-inference-research/scripts/benchmark/run_benchmark.py --model ingest_qwen3_32b --suite omniscience --new-run --server-mode --skip-speed-tests

# ingest_qwen3_coder_30b [ready]
/mnt/raid0/llm/epyc-inference-research/.venv/bin/python3 /mnt/raid0/llm/epyc-inference-research/scripts/benchmark/run_benchmark.py --model ingest_qwen3_coder_30b --suite omniscience --new-run --server-mode --skip-speed-tests

# math_qwen2_5_math_72b [blocked]
# blocked: /mnt/raid0/llm/epyc-inference-research/.venv/bin/python3 /mnt/raid0/llm/epyc-inference-research/scripts/benchmark/run_benchmark.py --model math_qwen2_5_math_72b --suite omniscience --new-run --server-mode --skip-speed-tests

# math_qwen2_5_math_72b_2 [blocked]
# blocked: /mnt/raid0/llm/epyc-inference-research/.venv/bin/python3 /mnt/raid0/llm/epyc-inference-research/scripts/benchmark/run_benchmark.py --model math_qwen2_5_math_72b_2 --suite omniscience --new-run --server-mode --skip-speed-tests

# minimax_m27_q8 [ready]
/mnt/raid0/llm/epyc-inference-research/.venv/bin/python3 /mnt/raid0/llm/epyc-inference-research/scripts/benchmark/run_benchmark.py --model minimax_m27_q8 --suite omniscience --new-run --server-mode --skip-speed-tests

# nemotron_cascade_2 [ready]
/mnt/raid0/llm/epyc-inference-research/.venv/bin/python3 /mnt/raid0/llm/epyc-inference-research/scripts/benchmark/run_benchmark.py --model nemotron_cascade_2 --suite omniscience --new-run --server-mode --skip-speed-tests

# qwen25_coder_32b_q4km [ready]
/mnt/raid0/llm/epyc-inference-research/.venv/bin/python3 /mnt/raid0/llm/epyc-inference-research/scripts/benchmark/run_benchmark.py --model qwen25_coder_32b_q4km --suite omniscience --new-run --server-mode --skip-speed-tests

# qwen35_122b_q4km [ready]
/mnt/raid0/llm/epyc-inference-research/.venv/bin/python3 /mnt/raid0/llm/epyc-inference-research/scripts/benchmark/run_benchmark.py --model qwen35_122b_q4km --suite omniscience --new-run --server-mode --skip-speed-tests

# qwen35_27b_q4km [ready]
/mnt/raid0/llm/epyc-inference-research/.venv/bin/python3 /mnt/raid0/llm/epyc-inference-research/scripts/benchmark/run_benchmark.py --model qwen35_27b_q4km --suite omniscience --new-run --server-mode --skip-speed-tests

# qwen35_27b_q6k [ready]
/mnt/raid0/llm/epyc-inference-research/.venv/bin/python3 /mnt/raid0/llm/epyc-inference-research/scripts/benchmark/run_benchmark.py --model qwen35_27b_q6k --suite omniscience --new-run --server-mode --skip-speed-tests

# qwen35_2b_q4km [ready]
/mnt/raid0/llm/epyc-inference-research/.venv/bin/python3 /mnt/raid0/llm/epyc-inference-research/scripts/benchmark/run_benchmark.py --model qwen35_2b_q4km --suite omniscience --new-run --server-mode --skip-speed-tests

# qwen35_2b_q6k [ready]
/mnt/raid0/llm/epyc-inference-research/.venv/bin/python3 /mnt/raid0/llm/epyc-inference-research/scripts/benchmark/run_benchmark.py --model qwen35_2b_q6k --suite omniscience --new-run --server-mode --skip-speed-tests

# qwen35_2b_q8 [ready]
/mnt/raid0/llm/epyc-inference-research/.venv/bin/python3 /mnt/raid0/llm/epyc-inference-research/scripts/benchmark/run_benchmark.py --model qwen35_2b_q8 --suite omniscience --new-run --server-mode --skip-speed-tests

# qwen35_397b_q4kxl [ready]
/mnt/raid0/llm/epyc-inference-research/.venv/bin/python3 /mnt/raid0/llm/epyc-inference-research/scripts/benchmark/run_benchmark.py --model qwen35_397b_q4kxl --suite omniscience --new-run --server-mode --skip-speed-tests

# qwen35_4b_q4km [ready]
/mnt/raid0/llm/epyc-inference-research/.venv/bin/python3 /mnt/raid0/llm/epyc-inference-research/scripts/benchmark/run_benchmark.py --model qwen35_4b_q4km --suite omniscience --new-run --server-mode --skip-speed-tests

# qwen35_4b_q6k [ready]
/mnt/raid0/llm/epyc-inference-research/.venv/bin/python3 /mnt/raid0/llm/epyc-inference-research/scripts/benchmark/run_benchmark.py --model qwen35_4b_q6k --suite omniscience --new-run --server-mode --skip-speed-tests

# qwen35_4b_q8 [ready]
/mnt/raid0/llm/epyc-inference-research/.venv/bin/python3 /mnt/raid0/llm/epyc-inference-research/scripts/benchmark/run_benchmark.py --model qwen35_4b_q8 --suite omniscience --new-run --server-mode --skip-speed-tests

# qwen35_9b_q4km [ready]
/mnt/raid0/llm/epyc-inference-research/.venv/bin/python3 /mnt/raid0/llm/epyc-inference-research/scripts/benchmark/run_benchmark.py --model qwen35_9b_q4km --suite omniscience --new-run --server-mode --skip-speed-tests

# qwen35_9b_q6k [ready]
/mnt/raid0/llm/epyc-inference-research/.venv/bin/python3 /mnt/raid0/llm/epyc-inference-research/scripts/benchmark/run_benchmark.py --model qwen35_9b_q6k --suite omniscience --new-run --server-mode --skip-speed-tests

# qwen35_9b_q8 [ready]
/mnt/raid0/llm/epyc-inference-research/.venv/bin/python3 /mnt/raid0/llm/epyc-inference-research/scripts/benchmark/run_benchmark.py --model qwen35_9b_q8 --suite omniscience --new-run --server-mode --skip-speed-tests

# qwen35_q4km [ready]
/mnt/raid0/llm/epyc-inference-research/.venv/bin/python3 /mnt/raid0/llm/epyc-inference-research/scripts/benchmark/run_benchmark.py --model qwen35_q4km --suite omniscience --new-run --server-mode --skip-speed-tests

# qwen36_27b_q4km [ready]
/mnt/raid0/llm/epyc-inference-research/.venv/bin/python3 /mnt/raid0/llm/epyc-inference-research/scripts/benchmark/run_benchmark.py --model qwen36_27b_q4km --suite omniscience --new-run --server-mode --skip-speed-tests

# qwen36_27b_q8 [ready]
/mnt/raid0/llm/epyc-inference-research/.venv/bin/python3 /mnt/raid0/llm/epyc-inference-research/scripts/benchmark/run_benchmark.py --model qwen36_27b_q8 --suite omniscience --new-run --server-mode --skip-speed-tests

# qwen36_q8_0 [ready]
/mnt/raid0/llm/epyc-inference-research/.venv/bin/python3 /mnt/raid0/llm/epyc-inference-research/scripts/benchmark/run_benchmark.py --model qwen36_q8_0 --suite omniscience --new-run --server-mode --skip-speed-tests

# reap_25b [ready]
/mnt/raid0/llm/epyc-inference-research/.venv/bin/python3 /mnt/raid0/llm/epyc-inference-research/scripts/benchmark/run_benchmark.py --model reap_25b --suite omniscience --new-run --server-mode --skip-speed-tests

# thinking_deepseek_r1_distill_qwen_14b_q6kl [ready]
/mnt/raid0/llm/epyc-inference-research/.venv/bin/python3 /mnt/raid0/llm/epyc-inference-research/scripts/benchmark/run_benchmark.py --model thinking_deepseek_r1_distill_qwen_14b_q6kl --suite omniscience --new-run --server-mode --skip-speed-tests

# thinking_deepseek_r1_distill_qwen_32b [ready]
/mnt/raid0/llm/epyc-inference-research/.venv/bin/python3 /mnt/raid0/llm/epyc-inference-research/scripts/benchmark/run_benchmark.py --model thinking_deepseek_r1_distill_qwen_32b --suite omniscience --new-run --server-mode --skip-speed-tests

# toolrunner [ready]
/mnt/raid0/llm/epyc-inference-research/.venv/bin/python3 /mnt/raid0/llm/epyc-inference-research/scripts/benchmark/run_benchmark.py --model toolrunner --suite omniscience --new-run --server-mode --skip-speed-tests

# worker_general [ready]
/mnt/raid0/llm/epyc-inference-research/.venv/bin/python3 /mnt/raid0/llm/epyc-inference-research/scripts/benchmark/run_benchmark.py --model worker_general --suite omniscience --new-run --server-mode --skip-speed-tests

# worker_summarize [ready]
/mnt/raid0/llm/epyc-inference-research/.venv/bin/python3 /mnt/raid0/llm/epyc-inference-research/scripts/benchmark/run_benchmark.py --model worker_summarize --suite omniscience --new-run --server-mode --skip-speed-tests

# Blocked roles are listed as comments above and in the JSON manifest.
