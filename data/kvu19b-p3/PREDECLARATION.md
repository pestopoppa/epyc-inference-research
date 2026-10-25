# PREDECLARATION kvu19b-p3 (RTG-57, KVU-19b-fold-c1-p3) - prepared 2026-10-25, NOT ARMED

Box (kv-unified-stack-rollout.md ~line 1063): "P3 B on the 27B against the folded champion: `-kvu` batched decode within ~5% of `-no-kvu`."
Class: indicative timing, block-timing-pairable (graphs ON, fresh process per arm, ABAB). Not a keep decision; observation that closes/keeps open the box.

## Binary
Resolved AT RUN TIME: `git -C /mnt/raid0/llm/llama.cpp rev-parse refs/heads/ak/champion/llama-cpp-ffc1bac82eec` -> sha; the store image is the ONE
`/mnt/raid0/llm/kernels/builds/gpu-*-<sha[:9]>-*` whose PROVENANCE.md says `status: CHAMPION`. None or several -> rc 6 (re-prepare). Today: dcc853b55 = C3 =
`gpu-20261023-dcc853b55-c3`. The sha, image and llama-batched-bench sha256 are recorded in RUN_RECORD.json. C4/C5 are NOT used until CAS moves the ref.

## Harness (pinned, unmodified originals; copies in vendor/, sha256 in SOURCES.sha256)
workspace-89's P3 v2 B phase (`p3_kvu_probe_v2.py bench_argv`): `llama-batched-bench -m Qwen3.8-27B-Q8_0.gguf -b 2048 -ub 2048 -t 8 -ngl 99 -fa on -ctk q8_0 -ctv q8_0
--no-mmap -dev ROCm0 {-kvu|-no-kvu} -npp 16384 -ntg 64 -npl 4 --output-format jsonl`, `-c` as bench_argv computes. No drafter: the P3 B phase is the no-draft
batched-decode comparison; DFlash2 is NOT loaded (the box names batched decode, no-draft arms on their own launch). Launch prefix
`env -u LD_LIBRARY_PATH ... HSA_OVERRIDE_CPU_AFFINITY_DEBUG=0 numactl --membind=3 -- taskset -c 184-191`; OMP stack env as the harness. `run_arm.py` is
the only new code (one fresh process per arm; the harness's A1 server phases are not run).

## Arms (each = one fresh llama-batched-bench process, HIP graphs ON, GGML_CUDA_DISABLE_GRAPHS removed from the child env and proved from /proc/pid/environ)
- K  = `-kvu`, GGML_CUDA_FA_SEQ_ROWS unset (skip / per-sequence layout default ON)
- N  = `-no-kvu`, same binary, same env
- S  = `-kvu`, GGML_CUDA_FA_SEQ_ROWS=0 (in-binary feature-off control; DIAGNOSTIC ONLY, never enters the rule)
Order (alternating, drift-robust): K1, N1, K2, N2, then S1. Resumable per arm (out/arms/<arm>.json is the done marker; a valid arm is never re-run).

## Measurand and rule (exact)
Metric: `speed_tg` (tokens/s) of the single jsonl row of llama-batched-bench, npp 16384, ntg 64, npl 4. Aggregation: mean over n=2 runs per arm.
  delta% = (mean(K1,K2) - mean(N1,N2)) / mean(N1,N2) * 100;  pair_i% = (Ki - Ni)/Ni*100.
  within-arm spread% = |x1 - x2| / mean(x1,x2) * 100 for K and for N.
box_result:
- UNRESOLVED if max within-arm spread > 3.0 %  (noise exceeds the edge's resolving power)
- WITHIN_5PCT   if delta >= -5.0 and both pair_i >= -5.0
- OUTSIDE_5PCT  if delta <  -5.0 and both pair_i <  -5.0
- UNRESOLVED otherwise (pairs straddle the edge)
References printed, not used in the rule: P3 v1 on production v10 = -15.23 %; workspace-ec expectation >= -5 %. S1 vs K/N printed as the SEQ_ROWS effect (one run).
Verdict COMPLETE only when K1,N1,K2,N2 are all valid (below); S1 invalid does not block COMPLETE (recorded). Else INCOMPLETE and the job resumes on the next grant.
Arm validity: rc 0, one row, not timed out, no guard trip, own-pid KFD VRAM non-zero in >= 5 samples DURING the run, every /proc/<pid>/task/*/status
Cpus_allowed_list within 184-191 in every during-run proof (first when VRAM is non-zero, then every 60 s), graphs ON proof, hip dlopen proof from our bin dir,
no failure patterns in the bench log. Stopping rule: after S1 (or STOP / deadline). No extra arms, no npp sweep, no retry of a valid arm.
GPU-LEAK-1: own-pid VRAM guard 60 GiB (arm killed, recorded as trip, job ends cleanly with ENDED).

## Decision values
box_result WITHIN_5PCT -> owner ticks the KVU-19b-fold-c1-p3 box with the per-arm table. OUTSIDE_5PCT -> box stays open, the table + S1 localise (SEQ_ROWS effect).
UNRESOLVED -> owner books one more ABAB pair; no claim. Indicative timing (no noise floor from a quiet-host repeat): never a keep/ship decision by itself.

## Belief capture
`[belief: cc]` ClaimTuple sketch: claim `kvu_batched_decode_vs_nokvu_delta_pct`, value = delta_pct, unit pct, lower_better false, n=2 pairs ABAB, scope Qwen3.8-27B-Q8_0
npl4 npp16384 ntg64 MI210 graphs-on, kernel = champion sha + llama-batched-bench sha256, window id, protocol none -> grades Judged/Located via claim_tuple.grade().
No native adapter (row 364: batched-bench wrappers are pre-hook); the owning session writes the tuple from VERDICT.json.

## Evidence (MEASUREMENT.md section 5)
On COMPLETE the job stages VERDICT.json, RUN_RECORD.json, PREDECLARATION.md, SHA256SUMS, README (5 lines) into evidence-staged/data/kvu19b-p3/ for the owning session
to commit to epyc-inference-research data/kvu19b-p3/. The job does no git operation.
