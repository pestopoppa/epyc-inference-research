# K28 Qwen3.6-35B-A3B GPU Prefill Trace Summary
Artifact: `/mnt/raid0/llm/epyc-inference-research/data/k28_gdn_perf/k28-qwen35moe-gpu-trace-verbose-pinned-20260720T112158Z`
Build commit: `93d945885`

## Prompt Throughput
| n_prompt | avg t/s | avg ns | backend | device |
|---:|---:|---:|---|---|
| 2048 | 2079.362 | 984917607 | ROCm | ROCm0 |
| 8192 | 1982.565 | 4132021574 | ROCm | ROCm0 |
| 32768 | 1650.802 | 19849744950 | ROCm | ROCm0 |

## Trace Attribution
`LLAMA_QWEN35_PREFILL_TRACE=2` emitted structural graph-node deltas, not wall-clock timing. Because `llama-bench` nulls model logs unless `-v`, this verbose run is the first usable trace artifact; the earlier non-verbose pinned run is a log-suppression negative control.

| trace n_tokens | groups | final graph nodes | GDN / linear-attn nodes | GDN / linear+FFN nodes |
|---:|---:|---:|---:|---:|
| 1 | 9 | 3727-3727 | 0.2450 | 0.1222 |
| 16 | 3 | 3727-3727 | 0.2450 | 0.1222 |
| 512 | 174 | 3727-3727 | 0.2450 | 0.1222 |

## Verdict
This does not overturn the existing K28 Phase 0 ceiling model. It confirms that GDN is structurally material inside linear attention, but it does not provide wall-clock attribution. K28 should remain default-off/post-promotion unless a direct profiler rerun or throwaway fused-loop prototype shows a materially higher full-model ceiling.
