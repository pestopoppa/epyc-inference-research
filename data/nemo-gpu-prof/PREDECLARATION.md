# PREDECLARATION nemo-gpu-prof (INF-65: NEMO-LILI-2 + NEMO-BOLT-PROF) - prepared 2026-10-25, NOT ARMED

Boxes (autokernel-champion-aggregate.md lines 65-66): NEMO-LILI-2 "profile one DFlash round on the MI210 (target batched 7-token verify vs drafter
encode/inject/draft decodes) to find why a round costs ~5x a plain token"; NEMO-BOLT-PROF "per-op rocprof of one Bolt decode step (IQ2_XXS expert MUL_MAT_ID,
NVFP4 Mamba in/out, SSM scan): why Bolt decodes ~53 t/s vs ~100 t/s for Qwen3.6-35B-A3B Q8". Profiling probe, NOT an AutoKernel launch (the 2026-10-24 AK hold stays; no
run.py / serial_run). Class: block-correctness = observation. There is NO threshold and NO keep/ship decision edge: the output is an attribution table.

## Binary and profiler
Champion store image resolved AT RUN TIME from `refs/heads/ak/champion/llama-cpp-ffc1bac82eec` (read-only git rev-parse): the ONE `kernels/builds/gpu-*-<sha[:9]>-*` with
PROVENANCE `status: CHAMPION`; none/several -> rc 6, re-prepare. Today dcc853b55 = C3 = gpu-20261023-dcc853b55-c3 (libllama carries nemotron_h_moe; verified by strings).
Profiler: AK's side-loaded rocprofv3 (rocprofiler-sdk 0.4.0, ROCm 6.2.0-66), sha256 c753449eb635ecb4d8be794e8b66439b200b252c157555920d260df5cbac767a, flags exactly the
arbiter's trace pass: `--kernel-trace --memory-copy-trace --stats --output-format csv`. HIP graphs ON (production shape; kernel durations are device-side). Launch prefix
`env ... HSA_OVERRIDE_CPU_AFFINITY_DEBUG=0 numactl --membind=3 -- taskset -c 184-191`. The knob `GGML_CUDA_DISABLE_GRAPHS` is proved in the binary by `strings` to a file
(not used here; recorded so a later counters pass may use it).

## Pass L (NEMO-LILI-2): Qwen3.8-27B-Q8_0 + Qwen3.8-27B-DFlash2-Q8_0, the :8083 launch record argv verbatim except binary, port 18431, slot-save path dropped, `-np 1 -c 16384`
(pool shrunk: a round's cost does not depend on the 393k pool; keeps VRAM far under the guard). One server, phases separated by 4 s idle:
warm (spec-on, 160 tok) | S = spec-on code request, temp 0, 160 tokens | P = same request with `speculative.n_max=0` (plain target tokens).
## Pass B (NEMO-BOLT-PROF): NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB.gguf (22.0 GB, IQ2_XXS experts) on the same champion image, NVIDIA recipe flags of
nemotron-bolt-champion-20261021/gpu_run.py (`--batch-size 2048 --ubatch-size 512 --parallel 1 -ngl 999 q8_0/q8_0 KV --flash-attn on --jinja -t 8`, ctx 16384, no mmproj, no drafter).
Phases: warm (48 tok) | M = code request, temp 0, thinking off, 96 tokens.

## Analysis (analyze_prof.py, model-free selftest passes before any load; pinned in SOURCES.sha256)
Bursts = dispatch runs separated by >= 2 s idle; the LAST bursts are the measured phases (L: S then P; B: M). Segments (one graph execution each) split at idle gaps >
max(50 us, 10 x the burst's median inter-dispatch gap); the first segment of a burst (prompt eval) is dropped. Per-kernel tables use the demangled name cut at the argument list
(template args kept: they carry the ggml type); category roll-up by name regex (ssm_scan, ssm_conv, gated_delta_net, moe_mul_mat_id, flash_attn, mul_mat_vec, mul_mat_q, quantize_q8_1, norm, rope, ...).
Reported, all as measured numbers (no pass/fail): L: plain-token busy ms (P, modal segment class), S segment classes (dispatches, count, busy ms, span ms), the target-verify class
(HEURISTIC = class with the largest mean busy time), drafter/non-target ms per round, busy ms per round and per generated token, round/plain busy ratio, wall tok/s of S and P from
server timings. B: modal decode-step class (>= 30 steps), busy ms and span ms per step vs wall ms per step from timings (coverage), top-25 kernels with ms/step, share, calls/step, category roll-up.
The target-vs-drafter split is a segment-size heuristic and is labelled so in the output; the primary, split-free numbers are busy ms per round and per plain token.
Pass valid iff: server healthy, kernel_trace.csv produced, >= 1 during-run proof with exe == champion llama-server and every /proc/<pid>/task/*/status Cpus_allowed_list within 184-191,
own-pid KFD VRAM non-zero in >= 5 samples DURING the run, server verified dead after SIGTERM (graceful: flushes the profile), analyser COMPLETE (L: >= 20 plain decode segments and >= 10 S segments; B: >= 30 modal steps).
GPU-LEAK-1: own-pid VRAM guard 60 GiB (server TERMed, pass INCOMPLETE, job ends cleanly with ENDED).
Verdict COMPLETE iff both passes valid. Passes resumable (pass-X/analysis.json with verdict COMPLETE is the done marker). Stopping rule: one successful run per pass; a failed pass is retried on
the next grant at most twice (attempt counter), then GAVE_UP with the failure recorded. No extra phases, no second sample, no sweep.

## Belief capture
`[belief: cc]` ClaimTuple sketch (observation, no decision): claims `dflash_round_busy_ms` (L), `plain_token_busy_ms` (L), `bolt_decode_step_busy_ms` (B) and per-category ms/step, unit ms,
lower_better true, n=1 (single traced request, no repeat, no noise floor), scope = model + champion sha + binary sha256 + rocprofv3 sha + graphs ON + MI210 + request shape, protocol none ->
Judged/Located via claim_tuple.grade(). Native adapter autokernel_profile.py / autokernel_aux_receipt.py need the AK receipt shape this probe does not produce (row 327/336), so not used.

## Evidence (MEASUREMENT.md section 5)
On COMPLETE the job stages VERDICT.json, RUN_RECORD.json, PREDECLARATION.md, both analysis.json files, SHA256SUMS and a 5-line README into evidence-staged/data/nemo-gpu-prof/
for the owning session to commit to epyc-inference-research data/nemo-gpu-prof/. The raw kernel_trace.csv stays in out/ (large; path recorded). The job does no git operation.
