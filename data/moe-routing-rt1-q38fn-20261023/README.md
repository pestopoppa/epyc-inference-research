# INF-72 RT-1 Qwen3.8-Flash-Next routing trace: offline analysis outputs (2026-10-23)

Traces: llama.cpp experimental/rt1-moe-trace-20261023 @ 42a0514db (v10 store libs, CPU), Qwen3.8-Flash-Next UD-IQ4_XS (qwen4exp, 48 layers, 512 experts, top-10, int16 ids),
3 workloads (prose wikitext-2 test, code v10-tree src + gguf-py, reasoning Fable-5 cot) x 512 windows of 512 tokens + 24 decode-leg windows each; 27 hash-verified shards.
Trace data: /mnt/raid0/llm/tmp/roi-prep-20261023/INF-72-RT1/results/ (not committed). Scripts: scripts/benchmark/moe_routing/.
All numbers are for THIS model: a PROXY for the RT-5 (Qwen3.6-35B-A3B frontdoor) and RT-6/RT-7 (DS41) targets, not their answer.

Files: validation.json (validate_traces.py), skew_report.* (RT-5, RT-5.1, cross-workload overlap), locality_report.* (IR_t, LRU reuse distance, co-activation),
residency_curve.* (RT-7), placement_replay_{same,cross}.* (RT-6; calibration on the same / the other domains).
