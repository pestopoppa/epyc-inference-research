# AXA-2 Hot Page-Cache Load / Lease Smoke

Artifact: `/mnt/raid0/llm/epyc-inference-research/data/gpu-mi210/axa2_qwen35_122b_hot_load_lease_smoke_20260719T065557Z`
Variant: hot page-cache 122B IQ2_M MI210 lease/load smoke, f16/f16 KV, c32768
Load-ready wall-clock: `7052 ms`
Request wall-clock: `315 ms`
Content: `READY`
Usage: `prompt_tokens=16`, `completion_tokens=2`, `total_tokens=18`
Cleanup: independent verification shows no KFD PIDs and no matching AXA server process.

Caveat: this is a hot page-cache/resident-lane acquisition smoke, not a cold-load measurement and not decision-grade until `P-GPU-1` is ratified.
