# KVU-12 redo (2026-10-21) np4 anchor noise floor

Copy of the redo's own measured np4 anchor-cell floor, made durable (was only in /mnt/raid0/llm/tmp/kvu12-redo-20261021/out/k4_verdict.json).
Byte-identical; sha256 in SHA256SUMS. Producer: research 2629df4ca, artifacts/np_context_kvu_cpu_study_20261020/ (Qwen3.6-35B-A3B-MTP Q8_0, production v10 cpu).
Floor (percent of anchor-cell median): decode sum 12.48, per-stream 11.93, TTFT 0.51. Cited by KVU-12.5 (FILLERS row kvu12-5-np8) as its predeclared max(2 x floor, 3%) min-effect basis.
