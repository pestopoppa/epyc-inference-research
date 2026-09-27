# gfx90a static register/ISA audit (INF03-REGAUDIT-1)

`gfx90a_isa_audit.py` is the standing, zero-GPU register and ISA audit for the gfx90a
(MI210) HIP kernels. It is also the **accept gate for any MMQ/FA register-affecting
candidate**: a candidate gets no timing until its `diff` against the incumbent passes.

It only reads binaries that already exist. It accepts a host library with a
`.hip_fatbin` section (`libggml-hip.so`), a clang offload bundle, or a bare gfx90a code
object (`.hsaco` / `.o`). The ELF, the kernel descriptors and the offload bundles are
parsed in Python. Metadata and disassembly come from ROCm's `llvm-readelf --notes` and
`llvm-objdump -d`; set `ROCM_LLVM_BIN` to use another toolchain. The tool never builds
anything and never touches a GPU.

It supersedes the intake-1823 dive scripts in
`/mnt/raid0/llm/tmp/dive-intake-1823/ourbuild/` (split_fatbin.py, analyze_kernels.py,
loop_analysis.py, agpr_rule_check.py).

## Commands

Run everything off the measurement cores:

```bash
T="taskset -c 72-79 python3 scripts/kernel_rnd/gfx90a_isa_audit.py"

# audit (default families: mmq, mmf, fattn_mma, fattn_wmma; --families all for everything)
$T audit /mnt/raid0/llm/kernels/production/gpu/libggml-hip.so.0.16.0 \
   --json v10.json --table v10.txt [--match REGEX] [--label NAME]

# readable views of a saved audit
$T table v10.json --families mmq --sort spill --limit 40      # sort: spill | acc | vgpr | name

# accept gate: candidate vs baseline. Exit 1 on any FAIL (or on any WARN with --fail-on warn)
$T diff incumbent.json candidate.json --families mmq,fattn_mma [--json diff.json] [--quiet-info]
```

The full library takes about 10 s on 8 cores: roughly 6,100 kernels in 135 code
objects.

## Per-kernel row (schema `epyc.gfx90a.isa_audit.v1`)

| group | fields |
|---|---|
| identity | `kernel` (mangled), `family`, `params` (MMQ type/`mmq_x`/`need_check`; FA DKQ/DV/ncols/softcap/mla; wmma D/ncols/nwarps/KQ acc type), `code_object`, `binary_sha256`, `row_id`, `self_sha256`, `stub`, `callees` |
| resources | `wg_max`, `reqd_workgroup_size`, `vgpr_total`, `arch_vgpr`, `agpr`, `accum_offset`, `sgpr`, `vgpr_spill`, `sgpr_spill`, `private_bytes`, `lds_bytes`, `tg_split`, `vgpr_alloc_granulated` |
| occupancy | `occupancy_waves_per_simd`, `occupancy_limiter`, `wg_per_cu`, `waves_per_wg`, `vgpr_alloc` |
| hot loop | `n_loops`, `n_hot_loops`, `hot` (counter maxima over the hot loops), `hot_loops_detail` (one entry per hot loop), `hot_spill_reloads`, `hot_spill_stores`, `hot_accvgpr_copies`, `hot_mfma_barrier_segments`, `hot_mfma_phases` |
| whole kernel | `kern` (the same counters over the whole body), `kern_spill_reloads`, `kern_accvgpr_copies`, `scratch_kind` |

The counters are: `mfma`, `dot`, `valu`, `accvgpr_read`/`accvgpr_write`/`accvgpr_mov`,
`priv_ld`/`priv_st` (split into `_off` and `_offen`), `global_ld`/`global_st`,
`ds_read`/`ds_write`, `s_barrier`, `waitcnt`, `waitcnt_vmcnt0`, `waitcnt_lgkmcnt0`,
`waitcnt_full` and `calls`.

## Heuristics

These are the tool's heuristics and their limits:

- **Private access.** A `scratch_*` op, or a `buffer_*` op whose resource is the
  private-segment buffer. At entry that buffer is `s[0:3]`, and it is followed through
  prologue `s_mov_b64` copies (for example to `s[64:67]`). Global `buffer_load`s through
  other descriptors do not count.
- **Spill reload.** A private `off`-addressed load in a kernel whose metadata reports
  `vgpr_spill_count > 0`.
  - With zero spills, private traffic is stack-object traffic (`scratch_kind=stack`).
    It is never counted as spills.
  - `scratch_kind=mixed` marks kernels where stack objects and spill slots coexist. For
    those, the reload count is an upper bound.
- **Loop.** A backward branch to a target inside the same symbol. Several back edges to
  one header are merged into one natural loop that runs to the far latch.
  - The dive took the shortest back edge, which undercounts FA-MMA bodies by about 2x.
    Their `s_cbranch_execz` early-exit latches return to the header partway through the
    body.
- **Hot loop.** One of the minimal loops that carry the kernel's maximum loop MFMA count
  (`v_dot` count if the kernel has no MFMA).
  - Many kernels have more than one hot loop. An MMQ stream-k kernel has two: the main
    tile loop and the fixup tile loop.
  - `hot_*` holds the worst case over the hot loops. `hot_loops_detail` holds each one.
  - Counts are static, taken over the loop body. For a body with internal branches they
    are an upper bound per iteration.
- **MFMA placement.** `hot_mfma_barrier_segments` is the MFMA count in each
  `s_barrier`-delimited phase. The loop is treated as cyclic, so the tail wraps into the
  first phase. `hot_mfma_phases` is the number of phases that contain MFMAs.
- **Occupancy.** Computed from the gfx90a VGPR file (512 per lane per SIMD, unified
  arch+acc, granule 8), SGPRs (800, granule 16), 8 waves/SIMD and 64 KiB LDS.
  - It uses **static** LDS only. Dynamic (`extern __shared__`) LDS is set at launch, and
    MMQ uses it, so the figure is an upper bound.
- **Stub.** A template instance compiled out as NO_DEVICE_CODE: no loop, no MFMA, and a
  body that only calls `no_device_code()`, or one of at most 16 instructions. Stubs are
  flagged and left out of tables, summaries and the gate.
- **MFMA-form regime.** Derived from the code object's `.comment`. `clang < 22` gives
  `mayneedagprs_rule_pre_159493`: the ROCm 6.2 rule under which kernels with
  `max_flat_workgroup_size <= 256` may get AGPRs and pay `v_accvgpr` copies. Newer
  toolchains give `llvm_159493_default`.

## Gate rules (`diff`)

Kernels are keyed by mangled symbol. When one symbol appears in several TUs, the worst
copy is used.

| severity | check |
|---|---|
| FAIL | `vgpr_spill` up, `hot_spill_reloads` up, `hot_accvgpr_copies` up, `occupancy_waves_per_simd` down, AGPRs appear (0 to >0) |
| WARN | `sgpr_spill` up (lands in VGPR lanes first), `agpr` up (already >0), `vgpr_total` up, `private_bytes` up, `lds_bytes` up, `kern_accvgpr_copies` up, hot-loop `vmcnt(0)`/`lgkmcnt(0)` drains up, hot loop lost, kernel removed, new kernel that spills or copies |
| INFO | improvements, and new clean kernels |

## Belief-kernel feed

`claim_projection(row, doc, ...)` returns `ClaimTuple` keyword sets. Each one is
OBSERVATION grade (`protocol_id=""`, `lower_better`, one static read), one per SC84/SC84a
metric: `vgpr_total`, `arch_vgpr`, `agpr`, `sgpr`, `vgpr_spill`, `sgpr_spill`,
`private_bytes`, `hot_spill_reloads` and `hot_accvgpr_copies`. `accum_offset`, the hot
counters, the MFMA/barrier segments, the MFMA-form regime, `source_commit` and
`flags_pragmas` travel in `extra`.

The claim identity is stated at audit time:

```bash
$T audit LIB --category BASELINE|CANDIDATE --source-commit SHA --flags "..." --json run.json
```

An audit written without `--category` projects **no** claims. That is the pre-hook case,
and the missing category is recorded as absent, never filled. Stub rows and absent
metrics are also skipped. The projection never grades. The test suite checks that the
root `claim_tuple.ClaimTuple` accepts its output. The root adapter and the ingest wiring
are SC84/SC84a. They are not done here.

## Tests

```bash
taskset -c 72-79 timeout 900 python3 -m pytest -q -p no:cacheprovider scripts/kernel_rnd/test_gfx90a_isa_audit.py
```

The fixtures are synthetic readelf/objdump text and packed descriptor/bundle bytes. One
integration test re-derives the Q8_0 J=64 datapoint from the frozen v10 library when it
is present, and is skipped otherwise.

The first audit (v10 production, and the v9 anchor against v10) is in
`data/gfx90a-isa-audit-20260926/`.
