# AKX-ALL-26 — width-invariance op probe (harness)

Harness only. It produces the per-cell records, the first-fail-k table and the prediction scorecard
for **AKX-ALL-26**; it does not ship any result. The cell set is derived from the frozen predicted-fail
list (`/mnt/raid0/llm/tmp/gpu-queue-20261020/akx-all-26/predicted_fail_list.jsonl`, 163 rows): every
predicted id yields records under the same id; the only addition is the positive control `PC-1`
(one per backend). Nothing else is added.

**Validation status (2026-10-20):** the CPU binary compiles cleanly against the champion CPU build.
The CPU dev validation (representative subset, k=1..9, 3 repeats, A/A, PC-1 must FAIL) has NOT been
run yet. Three 30-minute waits in a row for the assigned build claim (`--cpu-list 72-87` = region
q3) timed out (rc 75) because another job held q3 for the whole time. The HIP binary has not been
linked yet for the same reason. Do not run the official halves until the dev run passes:
`region-lock run --cpu-list 72-87 --role build --timeout-s 1800 -- <this dir>/devrun.sh`
(the script builds both binaries, smoke-tests the CPU one, checks that `--backend rocm0` on the CPU
binary aborts with exit 4, and runs the dev subset into `harness/devruns/cpu_subset`).

| file | role |
|---|---|
| `width_probe.cpp` | C++17, public ggml API only (`ggml.h`, `ggml-alloc.h`, `ggml-backend.h`, `ggml-cpu.h`), compiled out of tree |
| `run_width_probe.py` | stdlib driver: one probe process per id, explicit env per row backend, resume/shard, table + scorecard, merge of the two halves |

## Build (out of tree, under a build-role claim)

Both champion builds have `GGML_BACKEND_DL=OFF`: the backends are linked into `libggml.so`
(statically registered), so no `ggml_backend_load_all*` is needed (`--backend-dl-dir` exists only for
DL builds). `-lggml-cpu` is linked because the CPU run binds with `ggml_backend_cpu_init()`, which never
touches the global registry.

```bash
TREE=/mnt/raid0/llm/llama.cpp-experimental-champref-4348de400-20261006
SRC=<this dir>
H=/mnt/raid0/llm/tmp/gpu-queue-20261020/akx-all-26/harness
RL=/workspace/repos/epyc-orchestrator/scripts/region-lock
# CPU build (champion ref 4348de400)
CPUB=/mnt/raid0/llm/kernels/builds/cpu-20261006-4348de400/bin
$RL run --cpu-list 72-87 --role build --timeout-s 1800 -- /usr/bin/g++ -O2 -std=c++17 -I$TREE/ggml/include \
    $SRC/width_probe.cpp -L$CPUB -lggml -lggml-base -lggml-cpu -lpthread -Wl,-rpath,$CPUB -o $H/width_probe_cpu
# HIP build (link only; same ggml headers)
GPUB=/mnt/raid0/llm/kernels/builds/gpu-20261006-4348de400/bin
$RL run --cpu-list 72-87 --role build --timeout-s 1800 -- /usr/bin/g++ -O2 -std=c++17 -I$TREE/ggml/include \
    $SRC/width_probe.cpp -L$GPUB -lggml -lggml-base -lggml-cpu -lpthread -Wl,-rpath,$GPUB:/opt/rocm/lib \
    -Wl,-rpath-link,/opt/rocm/lib -o $H/width_probe_hip
```

## Run

### CPU half (one command)

```bash
/workspace/repos/epyc-orchestrator/scripts/region-lock run --cpu-list 48-95 --role build --timeout-s 0 -- \
  /usr/bin/python3 <this dir>/run_width_probe.py run --backend cpu \
  --build-bin /mnt/raid0/llm/kernels/builds/cpu-20261006-4348de400/bin \
  --binary /mnt/raid0/llm/tmp/gpu-queue-20261020/akx-all-26/harness/width_probe_cpu \
  --cases /mnt/raid0/llm/tmp/gpu-queue-20261020/akx-all-26/predicted_fail_list.jsonl \
  --out /mnt/raid0/llm/tmp/gpu-queue-20261020/akx-all-26/official/cpu --threads 48 --kmax 9 --repeats 3
```

Correctness work, not timing: any free CPU region is fine, but keep `--threads` > 1 (the predictions
assume nth > 1; 48 = production). The CPU FA split-KV chunking depends on the thread count, so the
thread count is part of the cell and is recorded in every record.

### HIP half (one command; run it inside the GPU hold)

```bash
/workspace/repos/epyc-orchestrator/scripts/region-lock run --gpu-quiet exclusive --role bench --timeout-s 0 -- \
  /usr/bin/python3 <this dir>/run_width_probe.py run --backend rocm0 \
  --build-bin /mnt/raid0/llm/kernels/builds/gpu-20261006-4348de400/bin \
  --binary /mnt/raid0/llm/tmp/gpu-queue-20261020/akx-all-26/harness/width_probe_hip \
  --cases /mnt/raid0/llm/tmp/gpu-queue-20261020/akx-all-26/predicted_fail_list.jsonl \
  --out /mnt/raid0/llm/tmp/gpu-queue-20261020/akx-all-26/official/rocm0 --threads 8 --kmax 9 --repeats 3
```

If the queue already holds `gpu-quiet` exclusively, run the inner `python3 ... run_width_probe.py`
command directly inside that hold. Before trusting HIP numbers, prove residency (CLAUDE.md "I invoked
the HIP build is not evidence of a HIP run"):
`/workspace/repos/epyc-inference-research/scripts/utils/verify_ggml_linkage.sh <binary> <build-bin>`
(the driver prints it; `--verify-linkage` runs it), AND every record carries `backend_bound` /
`device_description` — the probe aborts (exit 4) instead of falling back when `ROCm0` is not the
device it created. Sample VRAM during the run if you need residency evidence beyond that.

### Merge the two halves

```bash
python3 <this dir>/run_width_probe.py score \
  --cases /mnt/raid0/llm/tmp/gpu-queue-20261020/akx-all-26/predicted_fail_list.jsonl \
  --out /mnt/raid0/llm/tmp/gpu-queue-20261020/akx-all-26/official/combined \
  --inputs /mnt/raid0/llm/tmp/gpu-queue-20261020/akx-all-26/official/cpu /mnt/raid0/llm/tmp/gpu-queue-20261020/akx-all-26/official/rocm0
```

Resume: re-run the same command (ids whose part file ends in `row_done` are skipped; driver-level
ERR ids are retried only with `--rerun-errors`). Shard: `--ids A,B` / `--ids-file F` / `--shard I/N`
into the same `--out`, then any `run` or `score` rebuilds the outputs from all parts.

## Driver environment per predicted-row backend

The driver strips every inherited `GGML_*` variable and `LD_LIBRARY_PATH`, then sets
`LD_LIBRARY_PATH=<build-bin>` (CPU) or `<build-bin>:/opt/rocm/lib` (HIP) and:

| row backend | env | note |
|---|---|---|
| `cpu` | `GGML_IQK=1` | production CPU env (prediction assumption 3) |
| `cpu-native` | (none) | `GGML_IQK` unset |
| `cpu[GGML_IQK=1,GGML_IQK_Q8_0=1]` | both set to 1 | env variant |
| `graph` | — | emits a `NOT_PROBED` row (see caveats) |
| `rocm0` | (none) | GPU knobs at defaults (prediction assumption 2) |
| `PC-1` | `GGML_IQK=1` on CPU | |

The probe re-checks the env against the row backend and writes an `ERR` row on mismatch.

## Cell definitions (exact)

Common: operands are generated once per (id, sub-shape) from a fixed seed (`--seed`, default
`674854` = `0xA4C26`, recorded); each operand stream is keyed by `seed ^ fnv1a(id#sub-shape|operand)`,
so sharding/subsetting never changes operands. F32 activations ~ N(0,1); weights ~ N(0,0.05) in F32,
then `ggml_quantize_chunk` (after `ggml_quantize_init`) to the row type, multi-threaded over row
ranges. All i-quants (IQ2_XXS, IQ2_XS, IQ3_XXS, IQ3_S, IQ4_XS, …) get a deterministic positive
imatrix (`0.25 + U[0,1)`, seeded); K-quants and Q8_0 get none. Operands live in the backend's default
buffer (plain CPU buffer on CPU — not CPU_REPACK — device buffer on HIP).

For every k in 1..kmax ONE graph holds the batched op and k serial ops; all compared tensors are
graph outputs; the graph is computed `repeats` times (outputs are poisoned with 0xA5 before each
compute). Comparison is `memcmp` of identical byte layouts (tolerance 0); digests are SHA-256 of the
batched bytes and of the concatenated serial bytes.

| op | batched (width k) | serial i (i = 0..k-1) | compared layout |
|---|---|---|---|
| MUL_MAT | `mul_mat(W[ne00=K, ne01=M] T, view X[K, 0:k])` → Y[M,k] | `mul_mat(W, view X[K, i])` (contiguous column view of the same X) | k columns of M f32 |
| MUL_MAT_ID | `mul_mat_id(E[K,M,n_expert] T, view X[K, x_ne1, 0:k], view ids[n_used, 0:k])` | same with token i (n_tokens = 1), ids column i | k tokens of M·n_used f32 |
| FLASH_ATTN_EXT | Q stored [D,H,kmax] (token-major, as llama), op sees `permute(view Q[:, :, 0:k])` = [D,k,H]; K/V stored [D,H_kv,n_kv] (KV-cache layout) permuted to [D,n_kv,H_kv]; mask f16 [n_kv, k], query i sees positions j < n_kv-k+i+1; scale 1/sqrt(D), max_bias 0, softcap 0, no sinks, `GGML_PREC_F32` (as llama-graph) | same Q token i, mask row i (view of the same mask), same K/V | k queries of D·H f32 |
| GATED_DELTA_NET | `gated_delta_net(q,k,v,g,beta, s0, K=1)` over tokens 0:k | `gated_delta_net(token i, state_i)`, state_0 = s0, state_{i+1} = view of op i's state region | k tokens of S_v·H_v, then the final state S_v·S_v·H_v (batched final state vs LAST serial state) |
| PC-1 | `mul_mat(W F32[4096,512], X[:,0:k])` | `add(mul_mat(W[:, :K/2], X_i[:K/2]), mul_mat(W[:, K/2:], X_i[K/2:]))` (split-K: different summation order by construction) | k columns |

Shapes (prediction assumption 4; resolved from the id suffix):

| suffix | MUL_MAT W [ne00, ne01] | MUL_MAT_ID as [K, M, n_expert], top-n, X ne1 |
|---|---|---|
| `D27` | [5120,17408], [6144,5120], [17408,5120] (3 sub-shapes) | — |
| `M2048` | [2048,512] | [2048,512,256], top-8, X ne1 = 1 (gate/up) |
| `M512` | [512,2048] | [512,2048,256], top-8, X ne1 = 8 (down) |
| `F2560` | [2560,640] | [2560,640,512], top-10, X ne1 = 1 (gate/up) |
| `F640` | [640,2560] | [640,2560,512], top-10, X ne1 = 10 (down) |
| `ALL` (CPU) | all 7 above | the 4 MoE shapes |

FA: D=256, H=24, H_kv=4 (GQA 6), n_kv from the id (2048/7168/16384/65536), K/V type from the row
(`F16K/Q8_0V` = K F16, V Q8_0). GDN: S_v=128, H_k=16, H_v=48, plus H_v=32 for the F32 rows whose
shape names qwen35moe; state F32, or BF16 for `GPU-GDN-BF16state` (result dtype follows the state).
Gate form as qwen35 builds it: scalar g [1,H_v,T,1] = -exp(A_log)·softplus(alpha), beta = sigmoid,
q/k L2-normalised.

MUL_MAT_ID router: per token n_used distinct experts, seeded uniform (`--router random`, default).
Every expert referenced by any of the kmax tokens gets its own seeded weights; unreferenced experts
(never read) are byte copies of one referenced expert. `operands.first_collision_k` records the first
width at which two tokens share an expert (the CPU iqk MoE predictions depend on it). `--router
disjoint|collide` exist for follow-ups and are recorded; they are not part of the official run.

## Records

`cells.jsonl`: `run` (bound backend name + description, env, ggml version/commit), one `k` record per
(id, sub-shape, k) — `digest_batched[repeats]`, `digest_serial[repeats]`, `batched_repeat_stable`,
`serial_repeat_stable`, `width_differs`, `width_differs_by_repeat`, `first_differing_element`
(repeat, flat index, column, column kind, element in column, both raw hex and values),
`n_diff_elements…`, `max_abs_diff…`, `compute_ms` — and one `cell` record per (id, sub-shape):
`status` (OK / UNSTABLE / UNSUPPORTED / ERR / NOT_PROBED), `first_fail_k` (smallest k with
`width_differs`, null if none), `first_fail_k_stable_repeats_only`, `verdict`, unstable/unsupported/err
widths, `k1_identity_violation`, operand provenance. UNSUPPORTED (pre-construction: ne00 not a multiple
of the block size; or `ggml_backend_supports_op == false` at every width) and ERR are rows, never
omissions; a ggml assert writes an ERR row from the abort callback, and a dead process gets a
driver-level ERR row.

Exit codes of the probe: 0; 2 = PC-1 did not FAIL; 3 = an A/A repeat was unstable; 4 = usage/binding.

## Caveats (what the cell does and does not cover)

- Op level only. `GRAPH-GDN-builder` (llama graph builder choosing fused vs unfused GDN by width) is
  a graph-level prediction; it is emitted as `NOT_PROBED` and scored `not_probed`.
- k = 1 is the identical op (batched width 1 == serial op); a k = 1 difference therefore measures
  graph-position / buffer-placement dependence, and is listed separately (`k1_identity_violations`).
- One graph per k: batched and serial nodes share a graph, so backend graph-level scheduling (CPU
  tiny-op solo runs, HIP graph capture / fusion) applies to both, as in serving. A backend that fuses
  or reorders nodes differently for the two would show up as a width difference.
- Repeats re-compute the SAME graph and operands; they separate run-to-run nondeterminism from width
  variance. They do not re-quantize operands or rebuild graphs (a fresh process with the same seed
  reproduces operands bit-exactly).
- Real served shapes, synthetic values. Weight values are N(0, 0.05) quantized, not the served
  model's weights; activation statistics differ from served activations. A last-bit difference is
  value-dependent, so a PASS on these operands is evidence, not proof, of bit-equality on real data.
- CPU MUL_MAT_ID verdicts for collision-dependent types depend on `first_collision_k` of the seeded
  router; read it from the record before scoring the k.
- FA: no `n_seq` op param is set (`ggml_flash_attn_ext` leaves op_params[4]=0), so HIP
  `rows_are_seqs` is false — matching prediction assumption 2. Mask has exactly k rows (no padding).
- FA batched mask rows depend on k (query i of width k sees n_kv-k+i+1 positions), so the problem a
  given column solves differs across k; batched and serial always share the same mask bytes.
- PC-1 differs by construction (split-K), so it fails already at k = 1; it validates that the
  comparator/digest path detects a difference on each backend, not that a width effect exists.
- Weights are uploaded to the default buffer type; CPU_REPACK / extra buffer types are not exercised.
