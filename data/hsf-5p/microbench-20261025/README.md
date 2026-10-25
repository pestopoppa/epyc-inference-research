# HSF-5.P host KV mirror microbench, 2026-10-25 (re-run after G4 amendment)

Run `hsf-5p-microbench` (MI210, kv-host-mirror, fork build `14d0f7fa2`), ended 2026-10-25 13:48Z rc=0.
Rule: PREDECLARATION.md sha256 `9de08e8f6b6d6639b3f60ef3acd976099419c86da7b66ea9f57c9f0e48f23f70` (verified).
Amendment: AMENDMENT-G4-log-capture.md (instrument-only, all arms `--log-verbosity 4`).

Verdict: **FAIL**. G1 decode delta +4.17% (B 45.58 vs A mean 43.76 ms/token; A1 43.88, A2 43.64, A spread 0.55%) breaks the predeclared <=3% gate. G2 graphs reused PASS (737/742/737/737). G3 restore PASS. G4 mirror engaged PASS.

Files: VERDICT.json, launch.log, analyze.log, steps.txt, arms.jsonl (per-wave timings), bench12.jsonl (M1 bench).
Server logs (~200 KB each), vram.jsonl, slot bins and beliefs stay in /mnt/raid0/llm/tmp/f8-launchers-20261024/hsf-5p/gpu/out/.
