# HSF-5.P acceptance: PREDECLARATION (frozen 2026-10-24, before any build or compute)

Task: `handoffs/active/heterogeneous-slot-fabric-residency.md` HSF-5.P. Accept-test: the 3-step microbench of
`docs/design/hsf-5-host-kv-mirror-costing-20261024.md` ("Unmeasured" section). Prepared by workspace-f8. The gpu job refuses to run
if this file's sha256 differs from `PREDECLARATION.sha256`.

## Code under test

- llama.cpp fork branch `experimental/hsf-5p-host-kv-mirror-20261024` @ `14d0f7fa2b61c26f8edfacf91c300c37962e1d20`, cut from
  champion `dcc853b55` (`ak/champion/llama-cpp-ffc1bac82eec`). Worktree `/mnt/raid0/llm/worktrees/llama-hsf-5p-20261024`.
- Knob: llama-server argv `--kv-host-mirror SPEC` (absent = OFF). The A arms run the SAME candidate binary without the flag.
- Production v10 is not touched. This is not a promotion; it can reach production only through a v11 fold.

## Item 1: hsf-5p-build (cpu class, build role, no model)

- B0: CPU-only build (`-DGGML_HIP=OFF -DGGML_CUDA=OFF -DLLAMA_CURL=OFF -DGGML_NATIVE=OFF -DLLAMA_BUILD_TESTS=ON`) of target
  `test-kv-host-mirror`, run, then the build dir is deleted. PASS = exit 0 and the line `test-kv-host-mirror: all checks passed`.
- B1: HIP gfx90a build (house recipe gfx90a-house-v1) of `llama-server` and `llama-kv-mirror-bench` in `<worktree>/build-hip-hsf5p`.
  PASS = build ok, AND strings gate: `ggml_backend_kv_mirror_get_ops` in libggml-hip.so, `HSF-5.P kv host mirror ON` in
  libllama.so, `kv-host-mirror save:` in libllama-server-impl.so, control `graphs reused` present; negative control: none of the
  three gate strings in the champion image `kernels/builds/gpu-20261023-dcc853b55-c3` (the gate must be able to fail), AND
  `verify_ggml_linkage.sh` rc 0.
- B0 FAIL or B1 FAIL = the prototype does not compile or its host logic is wrong: fix on the branch, re-prepare (new CHANGE).

## Item 2: hsf-5p-microbench (gpu class, mi210_0 + host threads 184-191)

### Step 1-2 (M1, no model): `llama-kv-mirror-bench --node 3 --reps 3`

Host tier: anonymous mapping mbind(MPOL_BIND) to node 3, pinned+mapped in 1 GiB pieces with hipHostRegister(Mapped|Portable).
- step 1: D2H device -> mapped host at 4 KiB, 64 KiB, 400 KiB, 1 MiB, 156.9 MB, copy engine and kernel path, 3 reps (+1 warmup).
- step 1b: the per-step KV tail, spec ON shape (407,630 B): 128 fragments (ce / kernel / auto) vs one staged fragment.
- step 2: GatherFragmentsKernel (source-copied UMBP) H2D, rows of 2,176 B in runs of 1 / 16 / 512 tokens, 32,768 rows, interleaved
  destinations; kernel and copy-engine paths.
- Gate: every copy byte-verified (`ok=true` on every row). Rates are OBSERVATIONS (no rate gate): they fill the three "Unmeasured"
  rows of the HSF-5 doc. Reported shares use the GPU-ZOS-1a step times 139.62 / 59.08 ms (INDICATIVE, a DEBUG_TIMINGS build).

### Step 3 (M3): the :8083 shape, graphs ON, arms A1 / B / A2 / Bf, one load each, back to back in ONE grant window

- argv = the production launch record `/mnt/raid0/llm/epyc-orchestrator/logs/server_launches/8083.json` verbatim except binary,
  `--port 18385`, `--slot-save-path <out>/slots`; B adds `--kv-host-mirror k=6,node=3,mode=lag`, Bf adds `...mode=fresh`.
  The shape keys (-np 4, -c 393216, draft-dflash n-max 7, q8_0 KV, ROCm0, Qwen3.8-27B-Q8_0, --kv-unified, numactl --membind=3,
  taskset 184-191) are asserted; a changed record refuses (re-prepare). HIP graphs are left at their default (ON).
- Workload per arm (drive.py, fixed, identical across arms): warmup wave (64 tokens, excluded) + 6 waves of 4 concurrent greedy
  `/completion` requests (n_predict 384, ignore_eos, seed 1234, slot k = stream k); then the restore check on slot 0: prompt P
  (~4k tokens, deterministic) n_predict 96 -> `/slots/0?action=save` -> erase -> restore -> continuation (P + first output,
  n_predict 64, greedy).
- Metric: per wave, mean of the 4 streams' `timings.predicted_per_token_ms`; per arm, median over the 6 waves (>= 4 complete waves
  required).

Gates (analyze.py, selftest 7 cases):
- **G1 decode**: Abar = (A1 + A2)/2, spread = |A1 - A2|/Abar, delta = (B - Abar)/Abar. PASS iff delta <= 0.03 and spread <= 0.03;
  INCONCLUSIVE iff spread > 0.03 (the off arms' own noise exceeds the gate); FAIL iff delta > 0.03 with spread <= 0.03.
- **G2 graphs reused**: the last `graphs reused = N` server line per arm. PASS iff N_B / mean(N_A1, N_A2) >= 0.97. (The libllama
  graph-reuse counter; HIP-graph capture reuse itself is not logged at the production log level, so a broken capture is read
  through G1: KPF-18-S1.7g shows an un-captured decode moves the step far more than 3%.)
- **G3 restore greedy-equivalence**: continuation tokens of B and of Bf equal A1's, with A1 == A2 as the determinism control.
  A1 != A2 -> INCONCLUSIVE. B != A1 or Bf != A1 -> FAIL. (B is a lag save: it may stop at the last checkpoint and the
  continuation request replays the tail; Bf is byte-identical to a slot-file save by construction.)
- **G4 mirror engaged**: B's log has `HSF-5.P kv host mirror ON`, a `kv-host-mirror save:` line with tail_copies > 0 and
  0 <= replay_steps <= 6 (= k), and a `kv-host-mirror restore:` line with device_bytes > 0. Missing enable line -> INCONCLUSIVE;
  otherwise any miss -> FAIL.
- **Verdict**: PASS iff steps 1-2 verified and G1..G4 PASS; FAIL iff any FAIL; else INCONCLUSIVE (re-run in a new window).

Recorded, not gated: save/restore t_ms from the slot responses per arm (the handover cost the mirror removes; HSF-5 doc table 2),
B's save fallback_bytes (expected 0), own-pid VRAM peak per arm, host RSS.

## Decision this informs

HSF-5.P box: PASS = the prototype meets the DI-140 acceptance (fits the 3% decode budget with graphs ON, does not break capture,
restore is greedy-equivalent) and becomes a v11 fold candidate (with its value still bounded by handover frequency, HSF-3.1).
FAIL on G1 = the side-stream copy or the k=6 checkpoint wait costs more than estimated: re-cost with design B (D2D snapshot) or
k=13. FAIL on G3/G4 = a correctness defect in the prototype: fix before any fold.
