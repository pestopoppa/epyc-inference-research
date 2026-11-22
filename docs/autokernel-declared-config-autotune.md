# AutoKernel declared-config autotune for GPU knob sweeps (AMDSK-AUTOTUNE-CFG-1)

Status: design note, no code change. Source pattern: intake-2745 (amd/skills, `adopt_patterns`):
a kernel is tuned by a **declared config set**, a **retrigger key** (the call arguments whose change
invalidates the tuning), a **cached best**, and a **proof that each hint reached codegen** rather
than being silently dropped. Gate AMDSK-OCC-CALC-1 is done (`e5f5ce06e`):
`occupancy_from_notes()` in `scripts/kernel_rnd/gfx90a_isa_audit.py` reproduces the a10 VGPR lever
table (63->8, 78->6, 90->5, 125->4 waves per SIMD) from code-object metadata alone.

Scope: gfx90a / MI210, experimental tree only (`llama.cpp-experimental`; production v10 is never
touched). Every number stays an OBSERVATION; nothing here gates keep/promote.

## 1. What exists today (the sweep driver)

| Piece | File | Role | Limit for this pattern |
|---|---|---|---|
| Sweep loop | `scripts/kernel_rnd/kernel_sweep.sh` | reads TSV `label<TAB>variant-env`, runs each through `kernel_eval.sh`, ingests, prints Pareto | variants are **runtime env only**; a hint such as `waves_per_eu` is compile-time, so it cannot be expressed |
| Per-variant eval | `scripts/kernel_rnd/kernel_eval.sh` | correctness, A/B, rocprof; `BIN` / `BUILD_DIR` env select the binary | fenced off (`KERNEL_EVAL_ALLOW_DEPRECATED`); AK trusted evaluator is the live path. It has no post-build codegen check |
| Store | `scripts/kernel_rnd/kernel_store.py` | SQLite `runs`, `UNIQUE(label, ts, git_sha)`, correct-only Pareto (`_is_correct`, `pareto`, `best`) | natural key includes `ts`, so every run is unique: it cannot act as a cache key |
| Declared arms | `scripts/kernel_rnd/autokernel/loop/runtime_arms.py` | `RuntimeArmDeclaration` / `RuntimeArm`, `preflight`, `arm_state` (settled / pending / open against `surface_digest(anchor)`) | `KINDS` are runtime fields only (`env`, `threads`, ...); the closest existing declared-config mechanism, so the new schema mirrors it |
| Codegen evidence | `autokernel/codegen_summary.py` `gfx90a_isa_sidecar`, `retain_gfx90a_isa_sidecar`, `_toolchain_identity` | per-kernel `vgpr/agpr/sgpr/spill/scratch/lds/instructions` sidecar, create-only at `<store>/codegen/gfx90a_isa/<co_sha>.<sha>.json` | no occupancy field, no expectation check |
| ISA identity | `autokernel/device_isa_identity.py` `fingerprint`, `compare` | byte-identical normalized device code => `eligible_to_skip` | the "hint was a no-op" detector |
| Audit + diff | `scripts/kernel_rnd/gfx90a_isa_audit.py` `parse_notes_text`, `occupancy`, `occupancy_from_notes`, `analyze_kernel`, `compare_rows`, `run_diff` | static registers, occupancy, spills, hot loops, candidate-vs-anchor findings | none; it is the proof engine |

The sweep driver is therefore split: kernel_sweep.sh stays the **GPU-timing outer loop**; a new
**pre-GPU codegen stage** (CPU only) sits between "build" and "measure". The pattern is added as a
new declaration kind next to `RuntimeArmDeclaration`, not by overloading the env column.

## 2. Config-set schema

New module `autokernel/loop/compile_configs.py` (name tentative), schema
`epyc.autokernel.compile_config_set.v1`, same validation style as `runtime_arms.py`
(frozen dataclasses, `ArmDeclarationRefused`-style refusals, `MAX_CONFIGS = 16`, no duplicate
treatments by digest, mandatory `anchor` config with no hints).

```json
{"schema": "epyc.autokernel.compile_config_set.v1",
 "campaign_id": "...",
 "kernel": {"family": "mmvq", "symbol_match": "mul_mat_vec_q<(ggml_type)16, 1, true, false>",
            "quant": "IQ2_XXS"},
 "numerics_policy": "bit_exact_only",
 "configs": [
  {"config_id": "anchor", "hints": {}},
  {"config_id": "wpe6", "vehicle": "attribute_macro",
   "hints": {"waves_per_eu": {"min": 6, "max": 8}},
   "expect": {"occupancy_waves_per_simd_min": 6, "vgpr_spill": 0, "vgpr_total_max": 80},
   "numerics": "bit_exact", "rationale": "...", "evidence": "a10_iq2_vgpr_lever_20260812.md"}
 ]}
```

Hint vocabulary (each entry is `name -> value`; unknown names are refused):

| Hint | Value | Vehicle (how it reaches the compiler) |
|---|---|---|
| `waves_per_eu` | `{min, max?}` | `__attribute__((amdgpu_waves_per_eu(min,max)))` via a build macro or `-mllvm` |
| `launch_bounds` | `{max_threads, min_blocks?}` | `__launch_bounds__(max_threads, min_blocks)` template/macro arg |
| `flat_work_group_size` | `{min, max}` | `__attribute__((amdgpu_flat_work_group_size(min,max)))` |
| `num_vgpr` / `num_sgpr` | int | `amdgpu_num_vgpr` / `amdgpu_num_sgpr` |
| `unroll` | `{loop_id, factor}` | `#pragma unroll N` behind a `-D` macro keyed by `loop_id` |
| `lds_bytes` | `{static_bytes}` or `{dynamic_bytes}` | tile-size template parameter / launch argument |

Vehicles are limited to what the build recipe already owns (`controller/build_recipe.py` composes
`CMAKE_HIP_FLAGS`); a config that needs a source edit is a `source_candidate` and out of this schema.
`expect` is optional but, when present, **is the proof contract** (section 5): a config whose
`expect` fails is recorded `hint_unproven` and never goes to the GPU. Hints are compile-time, so
each config is a separate build in its own build dir (kernel_eval.sh `BUILD_DIR` / `BIN`); the build
takes the existing `build` CPU-region claim, no GPU.

## 3. Retrigger key args

The cache key is `sha256(canonical_json(...))` over exactly the inputs whose change can move the
best config. Not in the key: `label`, `ts`, run order, measured throughput (those are results).

| Key field | Source | Why it retriggers |
|---|---|---|
| `kernel_id` | `classify_family(symbol)` family + params (`gfx90a_isa_audit.py`) | different template instance = different register pressure (the a10 table varies 25..125 VGPR across instances) |
| `quant` / ggml types of src0, src1, dst | model GGUF tensor types | dequant state sets VGPR demand |
| `shape_class` | rows N, cols K bucketed, `ne11` batch bucket (decode 1..8 vs prefill), MoE `ids` flag | tile and occupancy optimum differs by batch regime; **not** raw context length unless the kernel reads it |
| `flags` | template flags (`true/false` wrapper bools) | the four IQ2_XXS variants differ 63..125 VGPR |
| `arch` + `wavefront_size` | `gfx90a`, 64 | register file model |
| `toolchain` | `_toolchain_identity`: compiler id/version, `CMAKE_HIP_FLAGS`, offload-arch | a compiler change can change which hints it honours |
| `anchor_isa` | `device_isa_identity.fingerprint(anchor)` | any source change that alters device code invalidates the cached best |
| `config_set_digest` | digest of the declaration | adding/removing a config reopens the sweep |
| `runtime_surface` | `runtime_arms.surface_digest(anchor)` | winner was measured under that runtime surface |

`arm_state` semantics carry over per config: `settled | pending | open` against the key.

## 4. Cached-best store

Location: `<ak_store>/autotune/best/<key_digest>.json`, create-only with a supersede pointer
(same discipline as `retain_gfx90a_isa_sidecar`: tempfile + `os.link`, never overwritten). Content:
`{key (all fields above), winner_config_id, proof_refs: [sidecar ref+sha], timing_row_ids,
pareto_basis, superseded_by}`. The timing rows themselves stay in `kernel_store.py` `runs`
(extend `raw` JSON with `config_id` and `key_digest`; no schema migration needed, then add an
index-by-key query). `kernel_store.py best` stays an OBSERVATION view; legacy rows are quarantined
(`legacy_unverified`), so it must not be the cache of record. Winner selection reuses
the existing correctness-first rule (`_is_correct` then Pareto), with one added disqualifier: a
config whose proof status is not `reached` cannot win.

## 5. Codegen proof per hint

Every proof reads the **retained code object** (not the build log or the source diff), via
`gfx90a_isa_sidecar` (notes) and `gfx90a_isa_audit.analyze_kernel` (kernel descriptor, disassembly),
compared with the anchor's sidecar. Per CLAUDE.md, a build path or flag string is not proof of
contents; the proof binds to `code_object_sha256` in the sidecar.

| Hint | Field read | Pass condition | Also record |
|---|---|---|---|
| `waves_per_eu {min}` | `.vgpr_count`, `.agpr_count` (unified), `.sgpr_count` -> `occupancy_from_notes()` -> `occupancy_waves_per_simd`, `occupancy_limiter` | `occupancy_waves_per_simd >= min` AND `vgpr_alloc <= floor(512/min)` rounded down to granule 8 (min 4: 128, 5: 96, 6: 80, 8: 64); no note field carries this attribute, so the effect is inferred from the register budget | `vgpr_spill`, `scratch_bytes` must be 0 (a10 finding 4: an occupancy gain bought with scratch is a loss). Clang ignores an unsatisfiable hint with a diagnostic only, so a missed `min` is `hint_ignored`, not a build error |
| `launch_bounds {max_threads}` | `.max_flat_workgroup_size`, `.reqd_workgroup_size` | `.max_flat_workgroup_size == max_threads` exactly | `wg_size` feeds `occupancy()` (workgroup granularity); `min_blocks` is proved through the `waves_per_eu` row, after confirming the HIP conversion on a mutant fixture |
| `flat_work_group_size {min,max}` | `.max_flat_workgroup_size` | equals `max`; bound of 1024 vs 256 also moves the VGPR cap (512/(max/64/4)) so read `vgpr_total` too | kernel descriptor `vgpr_alloc_granulated` (`parse_kernel_descriptor`) must agree with `.vgpr_count` rounded to 8 |
| `num_vgpr N` / `num_sgpr N` | `.vgpr_count` (arch part = `accum_offset` when AGPR>0), `.sgpr_count` | arch VGPR `<= N`; SGPR `<= N` | spills 0; `agpr_appears` finding from `compare_rows` |
| `unroll {loop_id, f}` | `hot_loops_detail[].insts`, `n_loops`, `n_hot_loops`, `hot.mfma`/`dot` | hot-loop instruction count scales by about `f` (tolerance declared in `expect`), loop count unchanged or reduced; `instructions` rises | spill and VGPR deltas (unrolling raises pressure): `compare_rows` FAIL findings disqualify |
| `lds_bytes static` | `.group_segment_fixed_size` | equals the declared bytes (computed from the tile parameters) | `occupancy_limiter == "lds"` flips on or off as predicted (64 KB per CU) |
| `lds_bytes dynamic` | not in the code object (set at launch) | proof is the dispatch's group-segment size from the rocprof kernel trace / launch capture (`race2_launch_capture`), then `occupancy()` is run with that size | verify the exact trace column name when implementing |
| any hint, no-op detection | `device_isa_identity.compare(anchor, candidate)` | verdict `identical` => the hint reached the compiler but changed nothing: status `no_effect`, GPU time skipped (this is AK-ISA-IDENT-1's eligibility test reused) | `indeterminate` is treated as differing, never as skip |

Proof statuses written to the sidecar-adjacent receipt: `reached` (predicted field moved, no FAIL
finding), `no_effect` (ISA identical), `hint_ignored` (ISA differs but the targeted field did not
move to the expectation), `regressed` (spill, AGPR appearance or occupancy drop per `compare_rows`),
`unproven` (llvm tools absent: `gfx90a_isa_sidecar` already returns `status: unavailable`; never
fabricated). Only `reached` proceeds to GPU timing. An optional second proof for attribute-style
hints is compiling the single TU with `--save-temps` and checking the function attribute in the IR;
it is slower and only needed when the register proof is ambiguous (hint already satisfied by the
anchor, i.e. anchor occupancy already >= min, which the calculator reports as `no_effect`).

## 6. Mapping onto the sweep driver

| Step | Where | Change when implemented |
|---|---|---|
| 1 declare + validate | new `loop/compile_configs.py`, modeled on `runtime_arms.RuntimeArmDeclaration.from_dict/preflight` | refuse unknown hints, duplicate treatments, missing anchor, over `MAX_CONFIGS`; `numerics_policy` as in runtime_arms |
| 2 key + cache lookup | new `autotune_key()` beside it; read `<store>/autotune/best/` | `settled` => skip sweep; `open`/`pending` => continue |
| 3 build per config | `kernel_eval.sh` `--build` path today builds once; sweep passes per-config `BUILD_DIR`/`BIN`; AK-native route uses `controller/build_recipe.py` + `source_candidate.py` | one build dir per `config_id`; linkage proof stays (`verify_build_linkage.sh`) |
| 4 codegen proof (CPU only) | `codegen_summary.retain_gfx90a_isa_sidecar` -> `gfx90a_isa_audit.occupancy_from_notes` + `run_diff`; `device_isa_identity.compare` | add `occupancy_*` fields to the sidecar kernel block, add `evaluate_expect(sidecar, anchor_sidecar, expect)`; table in section 5 |
| 5 GPU timing | `kernel_sweep.sh` loop body: only configs with proof `reached`; spec TSV gains a third column `config_id` (env column kept for runtime arms) | rows record `config_id`, `key_digest`, `proof_ref` |
| 6 cache best | `kernel_store.py` ingest (extra raw fields) + new create-only writer | winner per section 4; supersede on key change |
| 7 retrigger | sweep start recomputes the key; mismatch reopens | `anchor_isa` and `toolchain` are the usual triggers |

Sweep sizing (measurement sufficiency): the proof stage is zero GPU, so unreachable configs
are cut before any timing; a config set of N costs N builds and at most N' (reached) GPU rows. The
stop rule per set: stop when the cached best is `settled`, or when every config is `no_effect` /
`hint_ignored` (futility, no GPU spent).

## 7. Validation plan for the implementing row

1. Offline fixture: `_lever_notes(A10_LEVER)`-style notes for anchor and a `waves_per_eu=6` candidate;
   `evaluate_expect` must give `reached` for 78->(<=80) and `hint_ignored` when VGPR stays above the cap.
2. Mutant fixtures (as the C6 reward-integrity tests do): hint satisfied-by-anchor => `no_effect`;
   spill appears => `regressed`; llvm tools missing => `unproven`.
3. One CPU-only real build of a single TU with `waves_per_eu` on a served MMVQ instance, proof read from
   the produced code object; no GPU needed to prove codegen. GPU timing only afterwards, via the normal
   claim path.
4. Confirm on that build the two points this note marks as assumptions: the HIP `__launch_bounds__`
   second-argument conversion and the dynamic-LDS trace column.
