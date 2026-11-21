# GPU-resident speculative round for 27B Q8_0 + DFlash2 (GPU-RESIDENT-ROUND-2)

Owner: autokernel · 2026-11-21 · handoff: `epyc-root/handoffs/active/autokernel-loop.md` → GPU-RESIDENT-ROUND-2a..2e

## 1. Question and why

27B Q8_0 + DFlash2 (the served drafter) runs one speculative round in about 57 ms on the MI210. Each round
returns to the host several times: target verify, then host accept and rollback, then the drafter feature feed,
then host draft sampling, then the next verify. In production the EPYC cores and DRAM bandwidth are shared with
other tenants, so every host step in the round can stall behind a co-tenant. The question is:

> Can the whole decode loop of 27B + DFlash2 run on the GPU so that the host does no per-round control work, and
> what does that buy in (1) host CPU-seconds per generated token, (2) resilience of ms/round under a CPU-heavy
> co-tenant, and (3) GPU idle recovered?

This lever is judged on **CPU independence** as well as on speed. A slice that holds ms/round flat but cuts host
CPU-s/token, or flattens the contended arm, is a KEEP on its own terms.

## 2. What is already known

| Fact | Source |
|---|---|
| GPU idle per round: median 1.5 ms (2.7 %) greedy, 3.7 ms (6.7 %) temp 1; mean 6.4 % / 10.0 % including slot and HTTP tails | GPU-RESIDENT-ROUND-ATTRIB (rocprof re-analysis) |
| Host CPU busy 10.4 % of decode wall at temp 1 (about 5.9 ms CPU per 57 ms round) | census row `ak-gap-1-stoch-census-gpu-20261121` |
| Per-round host work: KV slot placement (`find_slot`), K/V index and KQ-mask inputs built in `set_input`, target→drafter feature gather across two contexts, sequential DFlash2 lattice selection, host RNG at temp > 0 | code read at `090d6c304` (GPU-RESIDENT-ROUND-1 box) |
| Device pieces already built: device argmax/accept (C17 `1b83c6967`), stochastic top-K accept (`4a9ca940d`, wide fast top-k `2c032ddf3`), GDN replay (`090d6c304`, KEEP: rollback becomes a pending-count update) | autokernel-loop.md |
| ROCm 6.2 and HIP 7.0 / 7.1 / 7.2 / develop have no conditional graph nodes | header check (operator brief) |

## 3. Architecture choice

### 3a. Persistent device loop (rejected for ROCm 6.2 / gfx90a)

A launcher kernel would poll a device work queue and launch the pre-captured graphs from the device.

- HIP has **no dynamic parallelism**: a kernel cannot launch a kernel on gfx90a (no device-side
  `hipLaunchKernel`, no `<<<>>>` from device code).
- `hip_runtime_api.h` in ROCm 6.2 defines `hipGraphInstantiateFlagDeviceLaunch = 4`, but the HIP device headers
  define **no device-side `hipGraphLaunch`** and no fire-and-forget or tail-launch stream. The flag has no usable
  device API behind it on this stack.
- The only persistent form left is a megakernel that runs every op of the round (about 2,650 kernels: 2,086
  verify, 561 drafter) inside one kernel, with grid-wide barriers between them. That means re-implementing the
  MMQ/MMVQ, flash-attention, GDN and top-k kernels as device functions under one occupancy budget. It is months
  of work and loses the tuned per-kernel launch geometry. Not pursued.

### 3b. K pre-captured fixed-shape rounds with device-flag early exit (RECOMMENDED)

- Every round input that the host builds today becomes a **device-computed** tensor in fixed-shape buffers sized
  for the maximum draft length (n-max 7 → 8 verify rows per slot), so one captured HIP graph per phase
  (verify, inject, draft) is valid for every round.
- The round reads a small **device control block** per slot: `n_past`, `n_accepted`, `pending` (GDN replay),
  `done`/`eos` flag, RNG counter. Accept kernels (C17 / stoch accept) write it; the next round's input kernels
  (mask, KV indices, positions, feature gather, lattice) read it.
- **Early exit without conditional nodes**: every kernel of a round checks the slot's `done` flag at entry and
  returns at once. A finished slot then costs only launch overhead for the rest of the batch.
- The host enqueues a batch of **K rounds** (K graph replays back to back on one stream), records an event, and
  goes back to HTTP. Accepted tokens go to a **device token ring** (pinned host-mapped memory); a host thread
  drains it asynchronously and streams tokens. The host checks stop conditions once per K-round batch.
- Cost of this choice: up to K-1 wasted rounds after EOS or a stop string per slot (bounded by K; the early-exit
  makes them cheap), and stop strings are detected up to K rounds late, which the emitter hides by truncating.

This gives **CPU-light**, not CPU-free: one enqueue and one event wait per K rounds instead of 4-6 host
round trips per round. With K = 8 and a 57 ms round, the host wakes about every 450 ms.

## 4. Device versions of today's host work

| Host work today | Device version | Slice |
|---|---|---|
| KQ mask built in `set_input_kq_mask` (n_kv × n_tokens cells) and copied H2D | mask = `log(step(p1 - p0 + 0.5))` from per-sequence cell positions; host writes n_kv × n_seqs floats | **2a (built)** |
| `find_slot` placement, K/V index inputs, token positions | fixed-capacity per-slot ring: each slot reserves n-max + 1 verify cells after its accepted prefix; a kernel computes K/V indices and positions from `n_past` in the control block; rollback = control-block update (cells past `n_past` are masked by position) | 2b |
| Target→drafter feature gather across two contexts (host D2H / H2D, encoder sync) | device copy between the two contexts' buffers (same device), row index = accepted count; fc encoder projection moved into the inject graph (ROUND-1 change 2) | 2c |
| Sequential DFlash2 lattice selection and host RNG at temp > 0 | lattice selection kernel per slot; Philox-4x32 counter RNG keyed by (seed, slot, round, position) | 2d |
| Per-round host enqueue and sync | K-round batch replay, device token ring, async emitter | 2e |

**Device RNG and correctness.** A device Philox stream cannot reproduce the host RNG sequence, so fixed-seed
identity with the host path is lost for temp > 0. Correctness at temp > 0 is judged by **distribution tests**
(SPEC-PIT.1: acceptance-rate difference CI within ±0.02, top-20 + other chi-square p ≥ 1e-3, KS on accepted
length), not by token identity. Greedy (temp 0) keeps bitwise token identity ON == OFF as its gate.

## 5. What the CPU still does

Tokenization and detokenization, HTTP and SSE, slot scheduling (admit, finish, cancel), prompt prefill
submission, stop-string matching on the drained token ring, and one enqueue plus one event wait per K-round batch.
Nothing on the host runs per round.

## 6. Expected gains

| Slice | GPU idle recovered / round | Host CPU saved |
|---|---|---|
| 2a mask | ~0 (mask build is mostly hidden behind the previous graph); ms/round expected flat | mask build O(n_kv·n_tokens) → O(n_kv·n_seqs): about 8× fewer host writes on an np 2 × 8-row verify, plus the same cut in H2D bytes. Expected tens of µs per graph; it matters as the first input removed, not as speed |
| 2b KV slot / indices | part of D→T (0.30 / 0.41 ms median) | `find_slot` + index inputs per graph |
| 2c feature gather | T→m3 + m3→m50 (0.76 / 0.98 ms mean greedy / temp 1) | encoder D2H/H2D and gather copies |
| 2d lattice + RNG | m50→D (0.49 / 1.93 ms mean) | host sampling, the largest temp-1 host phase |
| 2e K-round batch | the remaining launch gaps; ceiling = all per-round idle, 1.5 / 3.7 ms median (2.7 % / 6.7 %) | host busy 10.4 % of wall → target < 2 % (one wake per K rounds) |

The speed ceiling is the measured idle, 2.7 % greedy / 6.7 % temp 1 per round. The CPU ceiling is most of the
5.9 ms host CPU per round at temp 1. Under a CPU co-tenant the host phases stretch, so the contended arm is where
2b..2e should show their largest ms/round effect.

## 7. Measurement

Every slice row reports three numbers per arm (OFF / OFF2 / ON triples, fresh server each, A/A band ±1.5 %):

1. **ms/round** from `print_timing` (-lv 4): sum eval ms / sum (predicted_n / mean accepted length).
2. **Host CPU-seconds per generated token**: (utime + stime) of the server pid from `/proc/<pid>/stat`, sampled
   immediately before and after the timed requests, divided by tokens generated (CLK_TCK = 100, so use ≥ 384
   tokens per server to keep tick quantisation under 1 %).
3. **CPU-contended arm**: the same triple with a CPU-heavy co-tenant pinned to the GPU host-thread cores
   (`taskset -c 184-191`, SMT siblings 88-95, i.e. the cores the server's -t 8 threads use): `stress-ng --cpu 8
   --cpu-method matrixprod` started before the server warm-up and stopped after the last request. It reports
   ms/round contended / uncontended per arm; resilience = the ON arm's contended slowdown is smaller than OFF's.

Gates: greedy token identity ON == OFF (when OFF == OFF2); temp-1 distribution gates (SPEC-PIT.1) from 2d on.
Engagement witness per slice (startup line and content proof `strings` of the knob in `libllama.so`).

## 8. Risks

- **The scheduler leaves the mask chain on the CPU.** Input-only nodes can stay on the CPU backend; 2a pins the
  chain to backend 0 when it is a device. The smoke must confirm placement (`GGML_SCHED_DEBUG`, or VRAM delta),
  else the slice only moves host work around.
- **Graph-reuse churn.** A device-mask graph must reuse exactly when a host-mask graph would; 2a adds a mode
  check to `can_reuse`. A miss shows up as more graph builds (host CPU up), visible in the CPU-s/token metric.
- **Fixed-capacity KV ring (2b)** interacts with `-c` per slot, context shift and prompt cache reuse; those paths
  must fall back to `find_slot`.
- **Lost fixed-seed identity at temp > 0 (2d)**: the gate becomes distributional; a subtle sampling bias can
  hide in a weak test. Use ≥ 640k-draw oracle tests on the CPU kernel before the row.
- **Late stop detection (2e)**: up to K-1 rounds of wasted GPU per finished slot; K must be tuned against np and
  stop-string frequency.
- **ROCm graph replay cost**: HIP graph launch on ROCm 6.2 is itself ~tens of µs; K back-to-back replays are
  only CPU-light if hipGraphLaunch does not spin a host thread per graph.
- **Interaction with GPU-ZOS-2 overlap-process and ROUND-1**: those move the same host work onto a worker
  thread; 2c supersedes ROUND-1 change 2.

## 9. Slices, effort and order

| Slice | Content | Gate | Code est. | Row est. |
|---|---|---|---|---|
| 2a | device KQ mask (`LLAMA_DEVICE_KQ_MASK=1`) | none | 1 day (done: fork `980f7044d`, test fix `908353518`) | ~2400 s |
| 2b | fixed-capacity KV ring + device K/V indices + positions | 2a not FAIL | 3 days | ~2400 s |
| 2c | device target→drafter feature gather + fc encoder in inject graph | 2b | 2 days | ~2400 s |
| 2d | device lattice selection + Philox RNG (temp > 0) | 2c, SPEC-PIT.1 harness | 3 days | ~3000 s |
| 2e | K-round fixed-shape batch replay, control block, early exit, token ring | 2b-2d | 5 days | ~3600 s |

2a is first because it is self-contained, bitwise testable on the CPU, and is the first host input removed.
Its row measures CPU-s/token and the contended arm with ms/round as a non-regression gate.

## 10. Slice 2a as built

Fork branch `llama.cpp-experimental-gpuround2a-20261121` = `4a9ca940d` (C17 `1b83c6967` + stoch accept +
fast top-k) merged with replay `090d6c304` (`cb470168d`), then `980f7044d`:

- `src/llama-kq-mask-dev.h`: knob `LLAMA_DEVICE_KQ_MASK=1` and the op chain
  `get_rows(cpos, csel) → sub(tpos) → scale_bias(-1, 0.5) → step → log → cast(F16)`.
- `llama_kv_cache::kq_mask_dev_eligible` (causal, no ALiBi, no SWA, 1-D positions, not a tree ubatch) and
  `set_input_kq_mask_dev` (one row of cell positions per sequence of the ubatch, empty or foreign cells = 1e9).
- `llm_graph_input_attn_kv`: device-mask inputs, `set_input_kq_mask_any`, `can_reuse_kq_mask_dev`; used by the
  plain KV path and by `llm_graph_input_mem_hybrid` (the 27B hybrid GDN + attention path). The chain is pinned
  to backend 0 when it is a device.
- `tests/test-kq-mask-dev.cpp`: op chain vs an independent host-rule reference (bitwise, F32 / F16, unified and
  per-sequence streams), and with `KQMD_MODEL` an end-to-end 4-sequence speculative-like decode with rollback,
  ON vs OFF logits bitwise, unified and non-unified KV, with an engagement witness.

CPU test evidence (HIP_VISIBLE_DEVICES=-1, build-gpu at `908353518`): part 1 180 / 180 op cases bitwise;
part 2 ON == OFF logits bitwise, kv_unified 1 and 0, on Qwen3-0.6B Q8_0 (plain KV path, 39.5 M logits) and
LFM2.5-1.2B Q8_0 (hybrid path, 17.0 M logits), witness line seen; `test-gdn-replay-identity` still OK.
