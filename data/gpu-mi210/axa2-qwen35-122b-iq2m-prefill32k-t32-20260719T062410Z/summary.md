# AXA-2 122B IQ2_M 32K MI210 prefill follow-up

Artifact: `/mnt/raid0/llm/epyc-inference-research/data/gpu-mi210/axa2-qwen35-122b-iq2m-prefill32k-t32-20260719T062410Z`
Exit code: `143` (manual stop after no completed row)

No 32K row completed. The run used the current experimental HIP build with `-t 32`, `-p 32768`, `-n 0`, `-ctk q4_0`, `-ctv f16`, `-fa on`, and `-ngl 99`.

Observed failure mode: the process held MI210 VRAM and spent host CPU time, but emitted no stdout row and no stderr beyond ROCm init before manual stop after more than 14 minutes.

Cleanup: post_processes_empty=True, post_rocm_smi_no_kfd=True
