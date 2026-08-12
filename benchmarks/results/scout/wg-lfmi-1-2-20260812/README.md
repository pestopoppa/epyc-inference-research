# WG-LFMI-1 + WG-LFMI-2 (GPU arms) — LFM2.5-1.2B-Instruct

Date 2026-08-12. Grade: **SCOUT** (see *Grade* below). Kernel `production-consolidated-v9`
@ `0db32c06e3e550065b78311a6031ef3dd2c4f27c`, binary 10125, HIP tree
`/mnt/raid0/llm/llama.cpp/build-hip`. Scope: **the new model only** — gemma4 was NOT re-run;
its numbers are quoted from the 2026-08-12 record.

## WG-LFMI-1 — does the -Instruct variant reason? NO. Gate PASSES.

Template extracted from **both** GGUFs, byte-identical,
sha256 `f05bf4b967dc993bdc7a2fe6e43759ee218eb0eb340d68b063e1c4f8ad148176` (1783 chars):

- generation prompt is `<|im_start|>assistant\n`. **No `<think>` prefill**, conditional or otherwise.
- **No `enable_thinking` kwarg** — and none is needed. The only thinking-related branch is
  `keep_past_thinking` (default false), which *strips* `</think>` blocks out of **past** assistant
  turns. That is history hygiene, not a prefill. Contrast the abandoned LFM2.5-2.6B, whose template
  prefilled `<think>` unconditionally with nothing to turn off.

Control (the handoff's first trap — `llama-cli`/`llama-completion` silently ignore `-rea off`,
so a rendered prompt must be proven, not assumed): llama.cpp's own minja render via
`llama-cli --jinja -st -v` reports `"generation_prompt":"<|im_start|>assistant\n"` and
`"prompt":"<|startoftext|><|im_start|>user\n…<|im_end|>\n<|im_start|>assistant\n"` at **25 tokens**
for q1 — identical to the hand render fed through the raw `-no-cnv` path, also **25 tokens**.
`control_jinja_render_q1.err`. The `-rea` trap does not bite here because there is no reasoning
kwarg to be ignored, but the render was proven anyway.

Generated-token count, five WG-LFM-1 reference prompts verbatim, temp 0 / seed 42 / `--jinja -st`,
`n_predict 512`, every run stopping on EOS (not on the cap):

| prompt | Q4_K_M | Q8_0 | incumbent (gemma4, thinking OFF) |
|---|---:|---:|---|
| q1 capital of Japan | 3 | 3 | |
| q2 17*23 | 3 | 3 | |
| q3 first five primes | 14 | 14 | |
| q4 JSON object | 14 | 14 | |
| q5 marbles | 2 | 2 | |
| **total** | **36** | **36** | **33** |

**36 vs the 33-token bar = 1.09x — parity.** The abandoned 2.6B emitted 501 (15.2x); this is
13.9x fewer than that. Zero `<think>` strings in any of the ten outputs. The reset premise holds.

Wall time is **not** compared against the 2.97 s half of the bar: that was a 24-thread CPU
measurement and these ran on GPU. On GPU the five prompts cost 0.258 s (Q4) / 0.220 s (Q8) of
prompt-eval + generation.

### Caveat the token count cannot see: terse but not accurate — 3/5

Both quants answer q1/q3/q4 correctly and both get **both arithmetic items wrong**:
q2 `17*23` → 4931 (Q4) / 4935 (Q8), correct 391; q5 marbles → 360 (Q4) / 456 (Q8), correct 72.
Same two failures, different wrong answers, so it is model capability rather than a quant artifact.
Tokens-per-task is settled; **quality is now the open question** and WG-LFMI-3 is where it gets
answered.

## WG-LFMI-2 — GPU arms

`llama-bench`, `-ngl 99 -t 8 -r 5`, `-fa 0,1` swept (not assumed — the handoff records `-fa 0`
beating `-fa 1` for gemma4 on gfx90a). Pinned to the declared GPU host lane
`taskset -c 184-191` (`orchestration/stack_topology.yaml:220`). Three rounds with **rotated model
order** (R1 Q4→Q8, R2 Q8→Q4, R3 Q4→Q8) so monotonic drift cannot masquerade as a quant effect.
Each cell below is the mean of the three rounds' means; `sd` is across rounds.

| quant | fa | test | R1 | R2 | R3 | mean | sd | cv% |
|---|--:|---|--:|--:|--:|--:|--:|--:|
| Q4_K_M | 0 | pp512 | 14001.55 | 15204.48 | 13779.66 | 14328.56 | 766.64 | 5.35 |
| Q4_K_M | 0 | tg128 | 419.95 | 422.14 | 410.54 | 417.54 | 6.16 | 1.48 |
| Q4_K_M | 0 | tg512 | 418.21 | 417.21 | 407.64 | 414.35 | 5.84 | 1.41 |
| Q4_K_M | **1** | pp512 | 16479.19 | 13341.30 | 15217.28 | **15012.59** | 1578.93 | 10.52 |
| Q4_K_M | **1** | tg128 | 430.34 | 423.10 | 424.57 | **426.00** | 3.83 | 0.90 |
| Q4_K_M | **1** | tg512 | 432.37 | 423.63 | 437.54 | **431.18** | 7.03 | 1.63 |
| Q8_0 | 0 | pp512 | 16559.77 | 16608.09 | 19011.92 | 17393.26 | 1402.01 | 8.06 |
| Q8_0 | 0 | tg128 | 437.18 | 427.92 | 444.70 | 436.60 | 8.41 | 1.93 |
| Q8_0 | 0 | tg512 | 432.20 | 425.69 | 433.77 | 430.55 | 4.28 | 1.00 |
| Q8_0 | **1** | pp512 | 18435.52 | 18573.20 | 20380.93 | **19129.88** | 1085.62 | 5.68 |
| Q8_0 | **1** | tg128 | 445.62 | 427.10 | 446.16 | **439.63** | 10.85 | 2.47 |
| Q8_0 | **1** | tg512 | 448.22 | 433.43 | 444.55 | **442.07** | 7.70 | 1.74 |

**Max-opt is `-fa 1` for both quants** — unlike gemma4, where `-fa 0` won on the same GPU.
Peak VRAM (sampled during the run): **Q4_K_M 1.389 GB, Q8_0 1.905 GB**.

**Q8_0 beats Q4_K_M on GPU** (+2.5% tg512, +27% pp512), confirming and strengthening the recorded
Q4-vs-Q8 GPU inversion. Do not carry a CPU-derived quant choice onto this GPU.

### The number that actually decides — tokens per task

| | tokens/task | tg512 | time to answer |
|---|--:|--:|--:|
| LFM2.5-1.2B-Instruct Q8_0 | 36 | 442.07 | **81.4 ms** |
| LFM2.5-1.2B-Instruct Q4_K_M | 36 | 431.18 | 83.5 ms |
| gemma4-26B-A4B ORIG Q4_K_M | 33 | 100.54 | 328.2 ms |
| gemma4 crediting 1.44x MTP | 33 | ~144.8 | 227.9 ms |

**4.0x faster to an answer vs the incumbent's base GPU decode, 2.8x crediting MTP** — and unlike
the 2.6B, the token ratio does not eat the win. `llama-bench` cannot exercise MTP, so the 2.8x
column is the honest one to argue against; it still favours the challenger.

## GPU residency proof (`ldd` cannot establish this)

1. `verify_ggml_linkage.sh build-hip/bin/llama-cli build-hip` → PASS, all nine ggml/llama libs
   inside the HIP tree, `LD_LIBRARY_PATH` = `build-hip/bin:/opt/rocm/lib` (the ambient value,
   which puts the CPU `build/bin` early, was replaced not appended).
2. `ggml_cuda_init: found 1 ROCm devices … Device 0: AMD Instinct MI210, gfx90a` in every
   `bench_R*.err`; `backend = ROCm` in every result row.
3. **Sampled while the bench PID was alive** — `vram_R*.log`: VRAM 1.39 GB (Q4) / 1.90 GB (Q8),
   GPU use 95-99%, `kfd_clients=1` and the KFD client PID equal to the captured bench PID.
4. Live affinity, same samples: `Cpus_allowed_list = 184-191` on the bench PID itself. One
   sample (`vram_R3_Q4_K_M.log`, first line) read `0-191` — that is the pre-`exec` window before
   `taskset` applied, and every subsequent sample in every round reads `184-191`.

`verify_llama_cpp.sh` passed before any measurement (branch, commit, both binary digests).

## Grade — SCOUT, and precisely why

`scripts/preflight_canonical.py`, 2026-08-12 21:24Z, **verbatim**:

```
[1/5] uptime             PASS  uptime 0.0d ≤ 2.0d
[2/5] libomp             PASS  binary does not link libomp (built without OpenMP?)
[3/5] canonical_cmd      PASS  executor cmd starts with taskset -c 0-95 numactl --interleave=all, env carries OMP stack + LD_LIBRARY_PATH
[4/5] tripwire_bench     FAIL  Coder-30B-A3B Q4_K_M tg128 = 26.06 ± 6.93 t/s (target ≥28.0)
[5/5] freq_under_load    PASS  96/96 cores ≥2500 MHz under load (min 4291, max 4313, mean 4301 MHz)
PREFLIGHT FAILED — refusing to proceed
```

Record: `data/preflight/2026-08-12_212459.json`.

The gate refuses, so this is scout-grade; the failing leg is **`tripwire_bench`**. Uptime is no
longer the reason — that leg passes for the first time today. The tripwire is a CPU bench on cores
0-95, which a sibling agent's corpus re-embed was saturating during the check, and its stddev of
6.93 (27% of the mean) is the contention signature rather than a hardware regression.

**What was isolated and what was not.** Isolated: the GPU host lane 184-191 and its SMT siblings
88-95, reserved for this run and verified live on the bench PID. Not isolated: everything else —
the sibling held 0-87,96-183 and host load average was ~120 throughout. **The host was not quiet.**
The GPU arms should be largely insensitive to that (host threads uncontended, work on-device, and
tg cv is 0.9-2.5% across rounds), but the gate is the authority and the gate said no.

**Cheap re-run for decision-grade**: the whole GPU sweep above is ~2 minutes of wall time. Re-run
`bench_gpu.sh` for three rounds once the sibling's re-embed finishes and the tripwire clears, and
these cells become decision-grade with no methodology change.

## Replay

```bash
source gpuenv.sh
python3 render_lfmi.py                       # prompts (control: control_jinja_render_q1.err)
bash measure_lfmi.sh                         # WG-LFMI-1 token counts
bash bench_gpu.sh R1 "Q4_K_M:$M4" "Q8_0:$M8" # rotate the order for R2/R3
```
