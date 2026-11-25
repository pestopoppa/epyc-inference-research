# ak-27b-served-ub-arrival-sweep-gpu-20261125-r2 (durable copy)

Evidence for STACKCHG-27B-UB2048-20261125 (architect_critic :8083, -b/-ub 512 -> 2048).
Raw campaign root (local, not carried): /mnt/raid0/llm/tmp/ak-27b-served-ub-arrival-sweep-gpu-20261125-r2
(server logs under out/servers/, per-request results in out/results.jsonl).

- RESULT.json: verdict SCREENED; headline BEATS_ALL on per-user decode tok/s mean, total latency p50, total latency p95.
- spec.json, job.py: the pre-registered row and harness.
- vram_buffer_lines.txt: -lv 4 buffer lines from r0-i40-B2048-1795606347.log (first block) and r0-i40-B512-1795605721.log (second block).
