# AutoKernel loop — strategy

The loop's shape is normative and lives in
`docs/guides/agent-workflows/agent-loop-design.md` (epyc-root). This file is the
strategy the loop runs *inside* that shape: what to attack, what the hardware makes
possible, and what has already been settled.

## Prospective lineage capture (default off)

`lineage_capture.capture` can seal original patch and complete post-authoring
source bytes at `run.py:keep_the_diff`, before a lane reset overwrites its
`<mechanism>.<lane>.patch`. It is intentionally not called by the active loop.
Future authorization must first mint one durable attempt/capture ID and carry
that same ID into the native outcome row; only then can the patch-capture seam
call `capture(store, capture_id=attempt_id, parent_id=parent_attempt_id,
base_commit=base_head, patch_bytes=diff_bytes, source_files=source_bytes)`.
The source file set must be fixed before scoring. Historic patch filenames and
tree-level snapshot hashes do not establish this join; the offline exporter
must refuse them. The helper makes no candidate, benchmark, champion, or
promotion decision.

---

## The loop

```
when the champion changes (at most weekly otherwise):
    rocprofv3 the champion → ranked hotspots

each iteration, planner works in a worktree with the full toolbox:
    reads   champion · experiments.md · hotspots · hypotheses/inbox/
    probes  FREELY — llama-bench, rocprofv3, test-backend-ops -o OP --perf,
            llvm-objdump for VGPR/occupancy, env-flag sweeps. Nothing gated.
    forms   hypothesis H, backed by evidence it gathered itself

CRITIC PASS 1 on H, before any patch exists     · budget 3 rounds
    reject → reason returned VERBATIM; planner refines or regenerates
CRITIC PASS 2 on the committed diff, before the build   · budget 2 rounds
    reject → reason returned VERBATIM; H untouched, planner rewrites the patch

    build → affected-op units → independent reference → A/B alternating, n≥5
    keep → commit onto the champion branch
    else → negative, with mechanism and sample vector, into experiments.md
```

Every committed source keep (GPU or direct CPU) also gets a bounded diagnostic artifact at
`<store>/codegen/<champion-commit>.<backend>.<build-frame-sha256>.json`; its contents are embedded in that
keep's experiment row. The collector hashes up to eight standalone AMD
`.hsaco`/`.co` objects and, when ROCm `llvm-objdump` can disassemble them,
counts scalar/vector/matrix/memory instructions. CPU builds inspect only the
bounded installed `libggml-cpu.so` and allowlisted GDN/quant-dot wrapper symbols;
their x86 disassembly is a diagnostic sample, not proof that an edited helper
executed. A missing/stripped symbol reports unavailable rather than an inferred
instruction mix. This is **not** an additional keep gate. If the current HIP
library has no standalone code object, the producer can inspect a bounded sample
of gfx90a ELF objects in its embedded fatbin and bind each extracted object to
the candidate library hash and byte offset. This is not a complete fatbin census
or per-kernel attribution; unavailable bytes/tools remain explicitly unavailable.
Spills, occupancy, vectorization and CUDA PTX/SASS/CUBIN are likewise never
inferred from this MI210 diagnostic. A separate verified compiler/profiler
receipt is required before any of those become a mechanism claim.

---

## Porting gate order (AK-PORT-1/2)

For source changes, retain the actual diff and resolve its affected native op before
spending on a build. An unknown or shared source route is a gate refusal, never an
implicit `MUL_MAT` pass. Then compile the candidate, run `test-backend-ops -o` for
the affected op on the selected backend (host wiring and kernel-unit coverage),
compare with an independent reference where one is installed, and only then time
the candidate. Keep compile, op suite, and reference failures distinct in the
receipt. A fast but wrong candidate is discarded.

The experimental CPU `GATED_DELTA_NET` route has a deterministic F32 scalar
reference fixture with exactly representable outputs. The native CPU suite alone
compares CPU against CPU and is not independent evidence. It must be followed by
that scalar fixture. Two CPU IQK helper routes are admitted narrowly, each
requiring body-only edits to its named helper in `iqk_mul_mat.cpp` and the
nonempty native `MUL_MAT_ID` host/op suite under the resolved CPU recipe:

- `iqk_mul_mat_moe_rows`: trusted one-shot GDB witnesses prove `use_ref=true`
  on the reference and exact exported-helper entry in the candidate DSO for
  passing Q4_K/Q5_K cases. An independent scalar fixture compares their
  `MUL_MAT_ID` outputs at the fixed 40-row, two-token shape.
- `iqk_moe_fused_up_gate`: the native GLU selector currently has 0/0 CPU cases,
  so it is **not** a correctness witness. A separate fused up-gate graph fixture
  requires exact candidate-DSO helper hits and independent scalar agreement for
  both Q4_K and Q5_K, also at the fixed 40-row, two-token shape. The preceding
  `MUL_MAT_ID` suite checks host/op wiring; it does not prove fused execution.

The Q4_K/Q5_K dot-computation route is separate. For
`ggml/src/ggml-cpu/iqk/iqk_gemm_kquants.cpp`, target
`mul_mat_qX_K_q8_2_X4_T` and edit only the Q4_K/Q5_K-private dequantizers,
their private scale-unpack helpers, or that Q4_K/Q5_K-only template body.
The loop checks actual Git hunks against both the original and candidate
source; it refuses selector, shared Q6_K helper, header, sibling and other-file
edits. The native `MUL_MAT` and `MUL_MAT_ID` suite is followed by independent scalar
`MUL_MAT`, `MUL_MAT_ID` and fused up-gate comparisons for both quants across activation
widths 1–8. Each fused case requires a trusted hit on the matching template specialization
inside the candidate CPU DSO. A missing specialization, wrong output or
unproven path refuses timing. This route addresses the material dot work that
the v27 planner correctly found outside the older fused-helper-body boundary.

These receipts prove exact helper or specialization entry and those numerical
shapes, not every branch or production shape. The fused-helper and dot routes
supersede older abstentions only where their stated missing witness or edit
boundary is now satisfied;
historical nulls and abstentions for their actual measured mechanisms remain
evidence and must not be discarded. Other prospective IQK edits still refuse
before build: generic active-dispatch
logging cannot prove an edited quant function ran on a passing selected case.
x86 `quants.c` is shared by both
arms and is not admitted. Other CPU source families need a supported route
before admission. GPU matmul routes use the native CPU-reference suite;
the selected device block must report a nonempty passing count. On admitted GPU
source candidates, the gate checks that the selected `test-backend-ops` binary
supports seeded properties, then requires `AK_REF_V1` on every selected case.
Its compact metric receipt carries the suite's own per-case thresholds; missing
support or receipts are `oracle_unavailable`, not wrong-kernel evidence. This
does not admit CPU IQK source edits: their candidate-local CPU reference is not
independent. Do not infer a
universal cosine, PSNR, max-abs or MSE tolerance from this policy. Low-precision
comparison requires its own validated reference and threshold.

### Widened CPU source routes (operator decision 2026-09-26)

Nine more single-file CPU routes are admitted (`gates.CPU_SOURCE_ROUTES`; six below,
three more after them). Each needs the
named `target_symbol`, hunks inside the named bodies in both HEAD and the candidate, and
byte-identical headers (marker line through opening brace). A marker that recurs
elsewhere in the file never widens the boundary: markers are resolved inside the named
class, or before the disabled-build stub.

- **`dense_q8_tinyblas`**: `ggml/src/ggml-cpu/llamafile/sgemm.cpp`, target
  `tinyBLAS_Q0_AVX` (or `mnpack`/`gemm4xN`/`gemmMx4`/`gemm`).
  - **Scope:** the type-generic `mnpack`, `gemm4xN`, `gemmMx4` and `gemm` bodies of
    `class tinyBLAS_Q0_AVX`. This is the dense Q8_0 verify-batch GEMM (N=2..3 lands in
    `gemm4xN<2|3>`).
  - **Refused:** type-specific `load*`, `updot`, `denibble` and `bittobyte`, other
    classes, and added lines touching `.qs`/`.qh`.
  - **Op:** `MUL_MAT`.
  - **Reference:** `use_ref` does not bypass llamafile, so the native suite is not
    independent here. The independent evidence is two checks:
    - the scalar Q8_0 `MUL_MAT` fixture at widths 1–8;
    - a trusted GDB hit on `tinyBLAS_Q0_AVX<block_q8_0, block_q8_0, float>::mnpack`, in
      the candidate DSO, for the sole selected case.
- **`iqk_mmid_dispatch`**: `ggml/src/ggml-cpu/iqk/iqk_dispatch.cpp`, target
  `ggml_iqk_try_mul_mat_id`.
  - **Scope:** its implemented body. This covers the single-token slab path and the
    N>1 verify path:
    - thread-0 activation quantization;
    - the serial mapping pass and its barrier;
    - 1/nth per-expert stripes.
  - It is also the landing zone for the rowexact partition keeps.
  - **Op:** `MUL_MAT_ID`.
  - **Reference:** scalar Q4_K/Q5_K `MUL_MAT_ID` at widths 1–8, in alternating and
    single-expert modes. Plus a GDB entry hit and the Q4_K MoE ACTIVE marker on the
    selected n=4 case.
- **`iqk_dense_dispatch`**: same file, target `ggml_iqk_try_mul_mat` or
  `iqk_q8_0_enabled`.
  - **Scope:** those two bodies. The Q8_0 opt-in has defaulted off since aebb556b1.
  - **Ops:** `MUL_MAT` and `MUL_MAT_ID`.
  - **Reference:** scalar Q4_K/Q5_K/Q8_0 at widths 1–8, plus a GDB entry hit and the
    dense ACTIVE marker. Routing Q8_0 into iqk changes the activation quantization.
    The outputs are therefore not bit-identical to tinyBLAS: say so and price it.
- **2026-10-06 low-bit routes** (operator: "WE MUST improve lower quant performance").
  Numerics: `ppl_contract`, never bit-exact -- judged by `gates.ppl_contract_gate`'s
  four layers (test-backend-ops vs the use_ref reference at widths 1-5 for every type;
  wikitext2 |delta ppl| <= 0.5% vs the anchor and no cumulative growth vs the frozen
  production kernel; greedy agreement >= 0.98 on two >=4096-token wikitext prompts; a
  1024-token repetition canary relative to the anchor), ALL required, fail-closed, and
  re-run on the whole bundle before the champion of record folds. `cpu_repack_mmid`
  has no op-level oracle (test-backend-ops never uses the repack buffer) and is refused. Every file below has an
  x86_64 implementation above a top-level `#else` / `__aarch64__` arm redefining
  several of the same symbols; every route's window ends before that arm.
  - **`iqk_iquants_dequant`**: `ggml/src/ggml-cpu/iqk/iqk_gemm_iquants.cpp`, target
    `IndexHelperIQ3S`, `DequantizerIQ3S`, `EvenSignHelper`,
    `mul_mat_qX_K_q8_K_IQ_1`/`_IQ_N` or `iqk_set_kernels_iquants` (Q38FN seed 1).
    `IndexHelperIQ3S` is defined twice (a disabled `z_HAVE_FANCY_SIMD` arm and its
    `#else`); the `#else` (last in source order) is the one admitted.
  - **`iqk_legacy_iq4nl`**: `ggml/src/ggml-cpu/iqk/iqk_gemm_legacy_quants.cpp`, target
    the IQ4_NL dequantizers/unpackers or `iqk_set_kernels_legacy_quants` (Q38FN seed 3).
  - **`iqk_type_whitelist`**: `ggml/src/ggml-cpu/iqk/iqk_dispatch.cpp`, target
    `iqk_typeA_supported`, `iqk_weight_uses_q8_k` or `iqk_mmid_shape_supported` (Q38FN
    seed 3). The three constexpr whitelists a type must clear to reach `iqk_dense_dispatch`/
    `iqk_mmid_dispatch` at all; every admission needs a matching `static_assert`.
  - **`cpu_repack_mmid`**: `ggml/src/ggml-cpu/repack.cpp`, target
    `ggml_repack_get_optimal_repack_type` or `forward_mul_mat_id` (Q38FN seed 3). A type
    admitted into iqk by `iqk_type_whitelist` still needs repack to release it here, or
    it falls back to the generic un-repacked path.
  - **`iqk_kquants_q6_iq4xs_dequant`** / **`iqk_kquants_set_kernels`**: both
    `ggml/src/ggml-cpu/iqk/iqk_gemm_kquants.cpp` (Q38FN seed 6). The first widens the
    existing Q4_K/Q5_K dot admission to `DequantizerQ6K_AVX2`'s own kernel body
    (`mul_mat_qY_K_q8_2_X4_T`, a different template than Q4/Q5's, so it cannot collide
    with that route) and `DequantizerIQ4XS` (duplicated under the REAL `HAVE_FANCY_SIMD`
    macro, which is defined on this AVX512 host: the FIRST, fancy arm is the compiled one
    and the only one admitted); the second is the
    type-dispatch switch, `iqk_set_kernels_kquants`, a separate body entirely.
  - **2026-10-06 follow-up, served-shape layer (a)**: `served_shape_cases.py` adds a
    served-shape `test-backend-ops` corpus (`ServedShapeCase`, real `(k, m[, n_mats,
    n_used])` dims read off the lanes' own GGUF headers: DS41/`deepseek41` expert
    `(5120, 2304, 128 experts, 3 used)` and its dense shared-expert twin; Q38FN/
    `qwen35` dense FFN `(5120, 17408)`), at decode `n=1` and verify `n=2..5`, for every
    witness quant (`IQ3_S`, `IQ4_NL`, `Q4_K`, `Q5_K`, `Q6_K`, `IQ4_XS`), each held to a
    PER-CASE `max_nmse_err()` override (`test_mul_mat[_id]_served_shape`) derived from
    the ANCHOR's own measured NMSE on that exact shape/type times a small safety
    factor, capped at `SERVED_SHAPE_NMSE_CAP` (well below the generic flat `5e-4`) --
    never a shape-blind constant. `gates.check_served_shape_case_set`, wired into
    `ppl_contract_op_nmse` via `served_shape_manifest`, FAILS CLOSED (never skips, UNLIKE
    `cpu_fa_schedule`'s redundantly-probed FLASH_ATTN_EXT set) when the manifest or the
    binary's case-set literal is absent -- there is no independent probe covering these
    shapes the way `cpu_fa_reference`'s anchor-identity probe covers FLASH_ATTN_EXT.
    **Open operator action**: the manifest (`<store>/served_shape/manifest.json`) does
    not yet exist for any live store -- it is baked by running the independent
    `use_ref=true` reference on the ANCHOR at each `(shape, type)` pair, computing
    `served_shape_cases.tightened_nmse_bound` from the measured NMSE, and calling
    `served_shape_cases.write_manifest`; until that is done, `ppl_contract_op_nmse`
    fails closed on every ppl_contract candidate at this layer, by design.
  - **2026-10-06 follow-up, verify-width (n=2-5) coverage outside layer (a) --
    DOCUMENTED, not added**: layers (b)-(d) (`ppl_wikitext2`, `ppl_contract_coherence`,
    `ppl_contract_long_canary`) all drive `llama-perplexity`/`llama-completion`, which
    only ever decode one token at a time (`n=1`); plain prompt-processing batching
    (`-b`/`-ub`) is NOT the same code path as a speculative-decode VERIFY step (a draft
    proposing `n` candidate tokens the target accepts/rejects in one batched forward
    pass) -- so (b)-(d) exercise width 1 only, never 2-5, and no such verify-width
    end-to-end path exists in this tree's examples without introducing a draft model
    and an unverified new tool invocation (this session built/ran nothing, per the
    safety-review constraint, so a `llama-speculative`-based check was NOT added rather
    than guessed at). Verify widths 2-5 are therefore covered ONLY by layer (a)'s
    op-level suite above (both the generic sweep and, once baked, the served-shape
    corpus) -- at the per-op NMSE bound, never at a whole-model ppl/coherence/canary
    bound. A follow-up that wants (b)-(d)-level verify-width coverage needs a reviewed
    draft model and a `llama-speculative`-driven agreement check; this is a named gap,
    not a silent one.
  - **2026-10-06 follow-up, production reference loadability**: `_ppl_contract_layers`
    now runs `gates.check_production_reference_loads` FIRST (cheapest, before nmse),
    whenever `reference_build` is supplied. `ppl_wikitext2` reads a reference it
    cannot run as `ppl_r=None` and silently SKIPS the fixed-reference bar -- the
    candidate still gets scored against the anchor alone, which is indistinguishable
    from "the reference agreed" in the verdict reason. This preflight runs a 1-token
    `llama-completion` against `reference_build` + the lane's own model and fails
    closed, by name, if it cannot load it -- the exact failure a lane model on an
    architecture newer than the frozen production tree (DS41/`deepseek41`, Q38FN/
    `qwen35`, both well past the v10 freeze) would otherwise hit silently.
  - **Open item (NOT implemented -- flagged for operator clarification, not guessed
    at)**: "T0 short-prompt byte identity ... replace byte-identity with the
    token-agreement ratio used by layer (c)" from the 2026-10-06 follow-up list could
    not be located as a LIVE conflict in this loop: `gates.check_model_output_identity`/
    `check_model_identity_targets` (the loop's only short-prompt byte-identity check)
    fires ONLY for `witness.reference == "model_identity"` routes
    (`run.py` ~3579-3581), and `ppl_contract_gate` already raises if a route declares
    both `model_identity` and `ppl_contract` numerics (mutually exclusive) -- so no
    ppl_contract route reaches a byte-identity check through this path today. The one
    OTHER byte-identity-vs-tolerance mechanism found anywhere in `autokernel/` is
    `evaluator/correctness.py`'s `_derive_coherence`/`T0Policy` (the `t0.*` gate IDs),
    which has ZERO existing references to `ppl_contract` (`grep -rl ppl_contract
    scripts/kernel_rnd/autokernel` returns only `loop/`) and is a heavily
    self-locking anti-tampering module (`CoherenceTampering`,
    cross-reconciled determinism records) this review could not safely extend and
    fully re-verify without a live build/run. Widening it blind risked being the
    exact "garbage-generating change" this follow-up exists to prevent. An operator
    should confirm WHERE this check actually lives before it is implemented.
- **`float_tinyblas_plan`**: `ggml/src/ggml-cpu/llamafile/sgemm.cpp`, target `tinyBLAS`
  or `matmul` (class-qualified names resolve to the class they name).
  - **Scope:** only the `matmul` body of the float `class tinyBLAS` (F32/F16/BF16): the
    tile plan, i.e. which `mnpack<RM, RN, BM>` it calls and with what `SIZE_N`/`BN`.
    Narrow-M matrices get fewer jobs than threads today: DS41 `hc_mixes` (F16
    [20480, 24], 80 nodes per verify at N=3) runs as 24/8 = 3 jobs on 48 threads.
  - **Refused:** `mnpack`, `gemm_bloc`, the barrier-bearing `gemm`, `load`/`madd`/`hsum`,
    other classes, and added lines touching `A[`/`B[`/`C[`, `_mm*` or those helpers.
  - **Op:** `MUL_MAT`.
  - **Bit-exactness:** `gemm_bloc` accumulates each output element over k in one order
    whatever RM/RN/BM or thread computes it, so a plan-only edit is bit-exact. Say so.
  - **Reference:** `use_ref` does not bypass llamafile. The independent evidence is the
    scalar F16/BF16/F32 `MUL_MAT` fixture at widths 1–8 (40 rows on an 8-thread team, so
    inside the narrow-M regime; analytic bounds) plus a trusted GDB hit on the F16
    `tinyBLAS<...>::gemm<...>` in the candidate DSO for the sole selected case.
- **`cpu_norm_rowsplit`**: `ggml/src/ggml-cpu/ops.cpp`, target
  `ggml_compute_forward_rms_norm_f32` (or the dispatchers `ggml_compute_forward_rms_norm` /
  `ggml_compute_forward_rms_norm_mul_fused`, which resolve to the same body).
  - **Scope:** only the body of the `ggml_compute_forward_rms_norm_f32` template (F32).
    HEAD deals whole rows to threads. DS41's `hc_mixes` input norm is [20480, nt] with
    nt = 2–3 in serving verify (80 per graph, unfused: its consumer is the `hc_mixes`
    MUL_MAT), so 3 of 48 threads work. The mechanism is a within-row split for few, long
    rows. Every thread recomputes the FULL row sum of squares in HEAD's order, then scales
    only its own column segment. `get_rowcol_split` in `common.h` is the existing helper
    (see its use in `ggml_compute_forward_repeat_f32`). The same body serves the fused
    RMS_NORM+MUL, so the fused branch must split the same way or keep HEAD's code.
  - **In place** (`dst->data == src0->data`): keep the row split. Otherwise a thread
    scales its segment while another thread is still summing the row.
  - **Refused:** the header, the dispatchers, NORM/GROUP_NORM/RMS_NORM_BACK, and every
    other function. HEAD's `sum`/`mean`/`scale`/`eps` lines and the fused product line
    may be re-added verbatim, with any loop index. No other added code line may name
    `sum`, `mean`, `scale`, `eps`, `ggml_float`/`double`, `sqrt*`/`fma*`, a square
    `x[i] * x[j]`, `_mm*`/`__m*`/`GGML_F*`, `ggml_vec_*` other than
    `ggml_vec_scale_f32`/`ggml_vec_cpy_f32`, `#pragma` or `ggml_barrier`. RMS_NORM is
    tiny-solo eligible, and an in-op barrier on a solo node deadlocks. Comments go on
    `//` lines.
  - **Measured bound (2026-09-27, anchor-gen-001 ggml, 8 threads on cores 96-103):**
    the node is the serial double-add chain of the row sum: one [20480] row takes
    10.7 µs on one thread, and 12.0-12.3 µs for 2-3 rows on 8 threads. A correct
    one-task-per-thread split measured 11.4-11.9 µs, hot or with x and y evicted to
    DRAM, i.e. at most 0.4 µs (about 3%) of the node. Each split task recomputes that
    chain, so a split that deals a thread two (row, chunk) tasks doubles the node:
    `get_rowcol_split` with 3 rows on 8 threads measured 21.5 µs. Price a proposal
    against this bound. The 3.1-3.5% cycle share is the chain, and a bit-exact split
    cannot shorten it.
  - **Ops:** `RMS_NORM` and `RMS_NORM_MUL_ADD` (the fused-graph native suite).
  - **Bit-exactness:** the sum and the scale are computed exactly as HEAD computes them,
    only on more threads. The output must be bit-identical to HEAD. Say so; there is no
    tolerance to spend.
  - **Reference:** `use_ref` runs the same rms_norm body, so the native suite is not
    independent. The independent evidence is two checks:
    - `cpu_norm_reference`: 16 fixed cases on an 8-thread team, bit for bit against
      HEAD's arithmetic emulated without ggml, plus a float64 bound of 2^-20 relative.
      The cases are narrow (rows < 8, including [20480, 2|3], a 3-D walk, a strided
      view, in place ×16 repetitions, fused broadcast and full weights) and wide/solo.
    - trusted GDB hits in the candidate DSO on `ggml_compute_forward_rms_norm` (sole
      RMS_NORM case) and `ggml_compute_forward_rms_norm_mul_fused` (sole
      RMS_NORM_MUL_ADD case).
- **`cpu_graph_sync`**: `ggml/src/ggml-cpu/ggml-cpu.c`, target `ggml_barrier`,
  `ggml_cpu_node_is_solo`, `ggml_cpu_try_fuse_ops` or `ggml_graph_compute_thread`.
  - **Scope:** barrier implementation, per-node sync, tiny-solo selection and
    in-backend fusion.
  - **Ops:** 29 native suites: the 19 DS41 ops, the solo-eligible ops that list lacked
    (`SUB`, `DIV`, `SQR`, `SQRT`, `LOG`, `SIN`, `COS`, `FILL`, `DUP`), and `TOPK_MOE`, the
    whole-graph fixture that runs the MoE weight normalization (`GET_ROWS`, `SUM_ROWS`,
    `CLAMP`, `DIV`, SQRT_SOFTPLUS gating included) as one multi-node solo-run candidate.
    `UNARY` and `GLU` run their sub-op cases (`SILU`, `SWIGLU`, ...), except `EXP` and
    `EXPM1`, whose f32 cases fail on the anchor itself (overflow to inf in both arms). An
    empty suite is a harness refusal, never a verdict on the patch.
  - **Reference:** the full scalar quant suite. It checks numbers independently of the
    candidate's barrier. The native suites are not independent of the graph walk
    (`use_ref` only disables fusion), so they catch races and per-op numerics, not a
    deterministic walk defect that both arms share.
  - These edits must be bit-exact by construction. Name the ordering argument — EXCEPT a
    new in-backend fusion (below), which is judged in tolerance.
  - **New helpers (2026-10-03):** pure insertions at file scope that add NEW `static`
    functions/constants/tables, or `#include <...>` lines, are admitted alongside the
    body hunks (so a fused kernel can be a helper called from `ggml_cpu_try_fuse_ops`).
    No `#define`/`#if`/`#pragma` at file scope, no non-static globals, no edits to any
    existing function outside the named bodies.
  - **Fusion reference (2026-10-03):** the gate also runs `cpu_fusion_reference`: the
    exact DS41 `build_hc_mixes` graph (reshape → `ggml_rms_norm` without weight →
    `ggml_mul_mat` with F16 [20480, 24], nt = 1–3, plus odd teams/shapes) through the
    candidate's CPU backend, judged against float64 at 2^-10 × Σ|w·x|/rms (HEAD's unfused F16 path sits at <1% of that) (all-positive
    cases make that relative to |y|, so a dropped split-K slice fails), repetitions
    bit-identical to each other. A `rms_norm → hc_mixes` split-K fusion is TOL: say so.

### More routes (operator 2026-10-03: the loop must be able to author the seeds)

All three admit NEW file-scope `static` helpers and `#include <...>` lines (rules as
above) besides hunks in the named bodies.

- **`cpu_norm_numerics`**: `ggml/src/ggml-cpu/ops.cpp`, target
  `ggml_compute_forward_rms_norm_f32` (same body and targets as `cpu_norm_rowsplit`).
  - The rms_norm body is tried against `cpu_norm_rowsplit` FIRST: a bit-exact split
    still gets the bit-exact gate. A patch that changes the arithmetic (vector /
    multi-accumulator sum of squares, Fable seed 4) lands here.
  - **Refused:** `ggml_barrier` (solo nodes run on one thread) and `#pragma omp`.
  - **Ops:** `RMS_NORM`, `RMS_NORM_MUL_ADD`.
  - **Reference:** `cpu_norm_reference` in tolerance mode: the same 16 cases against
    float64 at 2^-16 relative (HEAD is within 5.5 u = 2^-21.5), repetitions bit-identical
    to each other, bit identity with HEAD NOT required (it is reported). Plus the GDB
    entry hits. This is TOL: say so and keep double (or ≥8-lane) accumulation — a single
    float chain over 20480 elements does not reliably meet 2^-16.
- **`cpu_mul_mat_body`**: `ggml/src/ggml-cpu/ggml-cpu.c`, target
  `ggml_compute_forward_mul_mat` or `ggml_compute_forward_mul_mat_one_chunk`.
  - **Scope:** those two bodies: src1 conversion/quantisation, `mm_batch1`, the
    `current_chunk` init and its barrier, the llamafile/iqk dispatch call sites and the
    chunk loop (Fable seed 6: per-thread redundant quantisation for small ne11; seed 9's
    prefetch may also live here). A per-eval cache of quantised src1 must not be keyed
    by a data pointer (`reward_hack_scan` refuses pointer-keyed caches) and must be
    invalidated every graph evaluation.
  - **Op:** `MUL_MAT`.
  - **Reference:** the scalar Q8_0/Q4_K/Q5_K/F16 `MUL_MAT` fixture at widths 1–8 (decodes
    stored bytes without ggml; `use_ref` runs this same body, so the native suite alone is
    not independent) plus a GDB hit on `ggml_compute_forward_mul_mat` in the candidate DSO.
- **`cpu_weight_placement`**: `src/llama-model-loader.cpp`, target `load_all_data` (or
  `load_all_data_parallel`, `llama_model_loader`).
  - **Scope:** those two bodies: PLACE the bytes being loaded (Fable seed 1: `mbind` of
    weight-tensor row quarters to NUMA node q via `syscall(SYS_mbind, ...)` /
    `get_mempolicy`, `madvise`, first touch). Values must not change. Reading
    `/sys/devices/system/node` is fine; `getenv`, `/sys/devices/system/cpu` and
    `sched_getcpu` are refused by `reward_hack_scan` — derive the node count from the
    kernel (`get_mempolicy`/`/sys/devices/system/node`), not the environment.
  - **Ops:** none — no op suite runs the loader.
  - **Gate:** `model_identity`: the anchor and the candidate llama-server, under the
    campaign launch (same argv/env/numactl prefix/port, only the build differs), each
    serve the first two frozen requests one at a time; the greedy completions must be
    byte-identical. An anchor that disagrees with itself makes the gate unavailable, not
    the patch wrong. Then the ordinary serving A/B measures the speed. This is PLACE:
    bit-identical by construction — say so.
  - The node-affine MoE slab half of seed 1 is an `iqk_mmid_dispatch` edit: propose it
    as a separate candidate after the placement keep.

These routes retire nothing measured. They correct one premise: the "220 GB/s read
ceiling" is contradicted by the same-boot C0 measurement under the identical launch prefix
(405–410 GB/s at 48 threads). The inbox brief carries the numbers.

### Structural routes (2026-10-04: CPU structural seeds)

The structural seeds remove O(100–1,000) graph nodes or recover idle threads; removing
O(10) barriers measured as noise. Three routes admit them. `cpu_graph_sync` is live again
(the C96 GLU/UNARY selector fault was repaired in `e29dca63`), so ignore any older note
that calls it unavailable.

- **`cpu_graph_sched`** (seeds 3 and 4B: co-group independent sibling nodes, expert
  groups): `ggml/src/ggml-cpu/ggml-cpu.c`, target `ggml_graph_plan`, `ggml_graph_compute`,
  `ggml_threadpool` or `ggml_compute_state`. The `cpu_graph_sync` targets resolve here too:
  that route is tried first, so a patch that fits it keeps its gate.
  - **Scope:** the four `cpu_graph_sync` bodies, plus the `ggml_graph_plan` body (size
    `wdata` for the concurrent pair) and the `ggml_graph_compute` body. Fields may be
    APPENDED at the tail of `struct ggml_threadpool` and `struct ggml_compute_state`
    (per-group chunk counters, group barriers). Existing fields may not be removed,
    reordered or edited. NEW file-scope `static` helpers (a group barrier, a group
    census) and `#include <...>` lines are admitted.
  - **Required:** fall back to group size 1, i.e. HEAD's sequential walk, when
    `cplan->use_ref` (or `params->use_ref`) is set. The reference arm of every native
    suite then runs the sequential schedule. A patch with no added line naming `use_ref`
    is refused.
  - **Refused:** `#pragma omp` anywhere but the `ggml_barrier` body. This is an OpenMP
    build: one `#pragma omp parallel` team per graph, and `ggml_barrier` is a team-wide
    `#pragma omp barrier`. A group that reaches a team-wide barrier another group never
    reaches deadlocks, so a sub-group needs its OWN group-local barrier (the dormant
    atomic path, parametrised per group) inside the existing team.
  - **Gate:** the `cpu_graph_sync` oracle (29 native suites, the scalar quant suite, the
    hc_mixes fusion reference). Then `model_identity` on this lane's target AND on every
    peer target the lane binding names (both models, when the binding is present). Each
    frozen request is served 3x by the candidate, prompt cache off, and the repetitions
    must be byte-identical. That is the race detector: a per-op suite cannot see a race
    that only two concurrent nodes produce. SCHED keeps every output element's
    arithmetic and changes only which thread computes it, so the result is BE-order by
    construction for per-row dot kernels. Name that argument.
- **`cpu_model_fused_op`** (seeds 1 and 6: Q38FN hyper-connection fused ops, packing
  same-input projections at load): target a qwen4exp builder, e.g.
  `llama_model_qwen4exp::graph::build_hc_mix` (also `build_hc_combine`, `graph`,
  `build_qkvz`, `build_layer_attn_linear`, `build_layer_ffn`, `load_arch_tensors`), with
  target surface `src/models/qwen4exp.cpp`. This file must change. The operator accepted
  model-specific kernels (2026-10-04). The template is DS41's
  `GGML_OP_DSV4_HC_PRE/COMB/POST`. The admitted files:
  - `src/models/qwen4exp.cpp`: hunks in those builder/loader bodies, plus NEW `static`
    helpers.
  - `src/llama-model.h`: field appends at the tail of `struct llama_layer` (a packed
    tensor).
  - `ggml/include/ggml.h`: `GGML_OP_*` enumerators appended directly before
    `GGML_OP_COUNT`, and NEW `GGML_API struct ggml_tensor * ggml_<op>(...);`
    prototypes.
  - `ggml/src/ggml.c`: NEW `struct ggml_tensor * ggml_<op>(` constructors, `static`
    helpers, tail appends to `GGML_OP_NAME`/`GGML_OP_SYMBOL`, and the two
    `static_assert(GGML_OP_COUNT == N, ...)` lines.
  - `ggml/src/ggml-cpu/ops.h` / `ops.cpp`: a NEW `ggml_compute_forward_<op>` prototype
    and definition, plus `static` helpers.
  - `ggml/src/ggml-cpu/ggml-cpu.c`: NEW `case GGML_OP_<appended>:` blocks in
    `ggml_compute_forward`, `ggml_get_n_tasks` and `ggml_graph_plan`. Each block is
    inserted at a case boundary, labelled only with the enumerators this patch appends,
    and ends in `break;`.
  - Nothing else changes: no existing op, field, enumerator or function.
    `src/llama-graph.cpp` (`build_moe_ffn`, every shared builder) is refused. A new
    name must not already occur in its file.
  - **Gate:** `model_identity`. Greedy completions must be byte-identical to the anchor
    on a target that serves a `qwen4exp` GGUF (this lane's own target or a peer). With
    no such target the gate is unavailable. When a changed path is outside the lane's
    exclusive paths (anything under `ggml/`, or `src/llama-model.h`), every peer target
    is also checked, and at keep time the lane binding's cross-target serving A/B runs as
    usual. A fused op that mirrors each sub-op bit-exactly (the fused decoder's design
    constraint) is what makes identity achievable. A TOL fusion fails this gate: say
    which you are proposing.
- **`cpu_graph_optimize`** (an alternative home for seeds 3/4B): target `graph_optimize`
  with surface `ggml/src/ggml-cpu/ggml-cpu.cpp`, which must change. Replace the CPU
  backend's `/* .graph_optimize = */ NULL,` slot with a NEW `static` hook that annotates
  co-group runs once per graph shape (a side table, or an `op_params` tail). The walk that
  consumes the annotation may change in `ggml-cpu.c` under every `cpu_graph_sched` rule.
  The `use_ref` requirement and the gate are those of `cpu_graph_sched`.

### Attention route (2026-10-04: long-context audit C2/C3)

The frozen requests run at a KV depth of about 300 tokens, where `FLASH_ATTN_EXT` is
1–5% of graph wall, so the ranked families will never put it first. Production is
different: on Q38FN `:8074`, 67% of decode wall is at contexts above 32k, and 55–68% of
per-token decode time scales with context. Decode/verify steps have N ≤ 5 query rows,
below the tiled path's 64, so each row is one serial `_one_chunk` walk over the whole
KV, threads split rows only, and the 12 query heads that share a Q38FN KV head each
re-read that KV. The problem is scheduling and GQA traversal, not bytes.

- **`cpu_fa_schedule`**: `ggml/src/ggml-cpu/ops.cpp`, target
  `ggml_compute_forward_flash_attn_ext_f16` (or `..._f16_one_chunk`,
  `ggml_flash_attn_ext_reduce_partials`, `ggml_compute_forward_flash_attn_ext`).
  - **Scope:** those four bodies (the FA dispatch, the per-row walk, the split-KV merge,
    the PREC switch), plus NEW file-scope `static` helpers (functions, or `const` objects
    with call-free initializers) and `#include <...>` lines. No `#define`/`#undef`/
    `#include "..."`/`#pragma` anywhere in the patch, no OpenMP in any spelling
    (`#pragma omp`, `_Pragma`, `omp_*`). The tiled body, the K/V conversion helpers,
    headers and `ggml-cpu.c` stay out (the per-row walk is admitted; keep its K/V reads
    as they are). Per-thread
    scratch is sized in `ggml_graph_plan` (not admitted): use the existing per-thread
    `wdata` budget, `MAX(prefill, decode) / n_tasks` bytes (about 290 KB per thread at
    D=256 and 550 KB at D=512), which holds G accumulators of DV floats.
  - **Admitted change:** work split and traversal ONLY. Every output row's reduction
    order stays exactly the anchor's: the same KV cells, in the same order, through the
    same `kq_vec_dot`/`expf`/`ggml_vec_mad_f16` sequence. Examples: walk the KV once per
    (KV head, row range) and update the G query heads that share it in that pass; deal
    (KV head, row) work instead of rows. A KV split with a merge is admissible only where
    it reproduces the anchor's order (the existing split-KV path's chunk boundaries
    under `GGML_FA_SPLIT_KV=1`); under the AK recipes' `GGML_FA_SPLIT_KV=0` the anchor's
    order is one serial walk, so a new split is not bit-exact. Leave the `use_ref`
    vec-only reference walk unchanged (the anchor-bit probe compares against the anchor
    build, so an edit there would move the reference with the candidate).
  - **Op:** `FLASH_ATTN_EXT` (the generic native suite).
  - **Gate, in order:**
    - `cpu_fa_reference`: the candidate's output must be BIT-IDENTICAL to the anchor
      build's on the `cpu_fa_longctx_v1` shapes (Q38FN D=256, 2 KV heads × GQA 12, kv
      8k/64k/128k, nb 1/5; DS41 D=512, 1 KV head × 64, sinks, kv 4k/8k/32k/64k, nb 1/3)
      plus layout and prefill guards, on a 7-thread and the recipe's team, under both
      `GGML_FA_SPLIT_KV` settings, each graph repeated 3x (races). This probe is compiled
      against each arm's own `libggml-cpu` and is the correctness gate today;
    - `model_identity` on the frozen requests, each served 3x by the candidate, on this
      lane's target and every peer target, and -- when the lane runs with
      `--longctx-surface` -- on the long-context manifest with the anchor's slot restored
      before every request;
    - the serving A/B (and, with `--longctx-surface`, decode at depth as the primary
      metric with the short surface as the no-regression gate).
    - NOT gates yet: the `cpu_fa_longctx_v1` `test-backend-ops` case set (selected by
      `AUTOKERNEL_CORRECTNESS_CASE_SET`) and the paired ABAB FA perf screen (geomean must
      drop ≥ 2.5%) run only once `test-backend-ops` carries the case set
      (`tmp/ak-cpu-fa-route-20261004/test-backend-ops-cpu-fa-longctx-v1.patch`, not yet
      applied to any tree). Until then both record a SKIP in the gate evidence.
  - **Bit-exactness:** BE by construction. Say so, and name the ordering argument: which
    loop order changed and why each row still sees the same sequence.

---

## What the instrument can actually resolve

Measured 2026-08-28, n=20 alternating A/A pairs, residency proven on 80/80
invocations (`artifacts/autokernel-aa-noise-floor/`):

| pairs | prefill p95 \|effect\| | decode p95 \|effect\| |
|---|---|---|
| 1 | 2.175% | 3.452% |
| 3 | 1.432% | 2.527% |
| 5 | **0.479%** | **1.502%** |
| 9 | 0.168% | 1.175% |
| 20 | 0.029% | 0.067% |

p95 of \|median effect\| over **every** C(20,k) subset — exhaustive, so it re-derives
exactly (`bench.MEASURED_FLOOR_PCT`). An earlier hand-written version of this table
quoted 0.753% / 1.848% at k=5, which no method reproduces from the raw pairs; it is
corrected here. The floors the loop **enforces** are 0.973% (prefill) and 1.544%
(decode) — deliberately *above* the measured row, because 20 pairs is a thin sample of
a heavy tail.

**4 of 20 pure-noise decode pairs already exceeded a 3% bar.** So: `n≥5`, and never
claim an effect smaller than the floor for the pair count used. Decode has heavier
tails and converges slowly; prefill is the cheaper surface to detect on.

**Serving floor across a source-only anchor change (DS41-C69, operator 2026-09-28).** The
matched serving floor is looked up as: exact anchor identity, then the champion-of-record's,
then the newest sealed floor of the SAME runtime recipe hash, carried read-only
(`serving_floor_provenance=carried_forward`) only when every later anchor-guard A/A on the
current anchor sits inside it. Otherwise the loop recalibrates (24 pairs, ~3 h); a runtime-recipe
change never carries (P-AK-SEARCH-1-A4), and `--no-floor-carry-forward` restores recalibration.

## The workload

`DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf` — n_embd 1536 (divisible by the
256-element K-quant superblock), **Q4_K ×169, Q6_K ×29**. That is production's
dispatch path at ~1 GB.

The superseded workload was `Qwen2.5-Coder-0.5B-Q4_K_M.gguf`, which despite its name
is **Q5_0 ×132, Q4_K ×12**: n_embd 896 is not divisible by 256, so llama.cpp fell
back silently and a month of screening measured a kernel production never dispatches.
`workload_contract.verify_workload` now refuses that class outright. **Never trust a
filename; census the tensor table.**

## The build recipe

`controller/build_recipe.HOUSE_GPU_RECIPE`, versioned, with every flag naming its
production counterpart. A flag that diverges without a stated reason is refused at
construction. `GGML_HIP_ROCWMMA_FATTN=ON` is not optional on gfx90a: the CMake
default OFF produces non-finite values at longer sequence lengths under `-fa on`, and
a short smoke test hides it because prompt length is the discriminator.

---

## Where to attack

Generate against the **live** profile of the current champion, not a frozen list.
Every accepted patch moves the distribution.

Standing seeds, carried because they are expensive to rediscover rather than because
they are ranked:

- **IQ4_XS / Q5_K and the 64-VGPR occupancy knee.** Every rung ≤64 VGPR (8 waves)
  decodes ≥90 t/s; both rungs above it (6 waves) decode ≤83 t/s *while 27–46%
  smaller*. IQ4_XS sits exactly on the boundary and is the fastest rung. Any IQ2/IQ3
  lever must carry an explicit VGPR target — a reduction landing at 70 buys nothing.
- **`iqk_gemm_1bit.cpp` and `iqk_flash_attn.cpp`** — vendored, audited clean
  2026-07-29, omitted from CMake, never staged.
- **The hypothesis inbox** (`<store>/inbox/*.md`, re-read every iteration): hypotheses
  arrive asynchronously — from the operator, or harvested from the backlog by
  `python3 -m autokernel.loop.seed`. Dropping a file in never blocks the loop.

## Measured gfx90a facts — check a mechanism against these before proposing it

These are receipts, not priors. A hypothesis that contradicts one is dead on arrival.

- **gfx90a LDS is 32 banks with 8 phase cliques** — *not* the CDNA3 answer (64 banks,
  2 phases of 32 lanes). Solved on our own silicon 2026-08-11 (372 bank / 6,048 phase
  dispatches; receipt SHA-256 `ae1d833c…`). Any swizzle constant lifted from CDNA3
  work — HipKittens' `>>7 <<3` among them — **does not transfer unexamined**.
- **`__builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8` does not exist on gfx90a.** Of the six
  `__builtin_amdgcn_*` intrinsics HipKittens uses across 67 headers, exactly this one is
  unavailable. Any fp8-MFMA mechanism is unimplementable here, not merely unmeasured.
- **Our MFMA fragment layout is already the standard one.** `ggml/src/ggml-cuda/mma.cuh`
  `tile<16,16>` (`:127,144` — `get_i = tid%16`, `get_j = 4*(tid/16)+l`, `ne=4`) is
  **bit-identical** to HipKittens' `rt_base`. Fragment-level techniques from that
  literature compose onto our existing tiles with **zero layout re-derivation** — this
  is an enabling fact, and it means "we would have to re-derive the layout" is not a
  reason to decline one.
  (Source for all three: `handoffs/active/agentic-rocm-kernel-authoring.md`.)

## Half of the prefill profile runs in a vendor kernel — BY OUR OWN DISPATCH

Measured on the contracted surface (pp512, DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M,
2026-08-28). Any hotspot whose signature begins **`Cijk_...`** is a **rocBLAS/Tensile
GEMM** from a vendor library, so the loop cannot edit that kernel's *body*:

**But it is reached by our own decision.** `ggml_cuda_should_use_mmq()`
(`ggml/src/ggml-cuda/mmq.cu:240`) chooses between MMQ and dequant+GEMM, and for
gfx90a/Q4_K it returns false purely because line :309 gates Q4_K/Q5_K at
`ne11 <= 256` while pp512 has `ne11 = 512`. One constant, in our tree, routes ~76%
of this surface into the vendor path. See seed `03-mmq-dispatch-threshold.md` —
changing WHICH kernel runs is available even when changing that kernel is not.

| share | patchable | kernel |
|---|---|---|
| 23.50% | **no** — Tensile | `Cijk_...MT64x64x64_MI32x32x8x1...` |
| 21.67% | **no** — Tensile | `Cijk_...MT160x128x64_MI32x32x8x1...` |
| 15.06% | yes | `dequantize_block_q4_K<__half>` |
| 10.94% | yes | `flash_attn_ext_f16<128, 32, 4, 64, float, false>` |
| 9.32% | yes | `convert_unary<float, __half>` |
| 4.14% | **no** — Tensile | `Cijk_...MT64x128x64...` |

**~51% vendor, ~47% ours.** So the addressable ceiling on this surface is roughly half
the device time, and an effect must be scaled accordingly: a 10% win on
`dequantize_block_q4_K` is 1.5% of the surface, which is above the 0.973% floor but not
by much.

**Read the shape, not just the rows.** Prefill here is a *dequantize-then-vendor-GEMM*
path: `dequantize_block_q4_K` -> `convert_unary` -> Tensile GEMM is **75.7%** of device
time. The weights are unpacked to f16 and multiplied by a library kernel. That is the
structure a mechanism has to engage with, and the biggest patchable single target on
this surface is the dequant itself.

Note what this implies about the decode surface by contrast: `tg128` dispatches
`mul_mat_vec_q` — quantized matmul, entirely our source, no Tensile at all. Decode has
a far higher patchable fraction and a higher noise floor (enforced 1.544% vs 0.973%; measured 1.502% vs 0.479%).
Neither surface is wrong; they are different bets, and the choice is the operator's.

## Known hazards when patching the dequant path (learned the expensive way)

Run 9 aimed correctly at the two biggest patchable slices — `dequantize_block_q4_K` (15.06%)
and `convert_unary` (9.32%) — and lost both to the same two failures. Answer these before
authoring, not after the build:

- **`dequantize.cuh` is a SHARED header.** It is also included by `cpy.cu`, which does not
  declare `ggml_cuda_cast`. A new kernel in that header calling `ggml_cuda_cast<half>` compiles
  fine where you are looking and then fails the whole build with undeclared-identifier errors
  from a translation unit you never opened. Check every includer of a header before adding a
  call to it, and prefer putting new code in the `.cu` that needs it.
- **The dequant kernels sit on the MUL_MAT path.** Two separate run-9 attempts died on
  `MUL_MAT failed on ROCm0`. The anchor passes 1139/1139, so the failure was the patch, not the
  baseline. A change to dequantize/convert changes how every quantized matmul reads its
  weights: a hypothesis here must say what keeps the numerics identical, not only what makes
  it faster.
- **Claiming a path you did not change is a transient, not a patch.** The worktree is the
  ground truth and is reset to the champion before every iteration, so "authoring returned no
  changed paths" means the edit never landed. Re-read the file before reporting paths.

## A DERIVED marginal is not measured against the same bar

Two effects measured against a common anchor give an exact marginal:

    tput(B)/tput(anchor) / [ tput(A)/tput(anchor) ] = tput(B)/tput(A)

The arithmetic is exact. The UNCERTAINTY is not the same. Each input carries the
floor's p95; their ratio carries roughly **sqrt(2) times** it, because two independent
errors compose. Comparing a derived marginal against the ordinary floor asks it to
clear a bar calibrated for a directly measured quantity, and rejects real gains.

This is not hypothetical: `akm-q4k-q8-sum-sidecar` was demoted from the champion on
exactly that mistake. Its marginal was +1.088% against a 1.188% floor -- but the
correct interval is +/-1.680%, so the honest verdict was UNDEMONSTRATED, not refuted.
It may be worth up to +2.8%.

**So: if you want a marginal, MEASURE it.** With the anchor advancing, a candidate is
already compared against the current champion, and that number is direct. Only infer a
marginal from two cumulative figures when you cannot measure it -- and when you do,
widen the bar to sqrt(2) x floor and say that you did.

## What the correctness gate does NOT catch

`test-backend-ops` is the gate, and it is necessary rather than sufficient. A real
corruption has already passed a green test suite here: `b8ad9d292` incidentally
reclassified Q2_K/Q3_K activations `Q8_2_X4` → `Q8_K`, which engaged never-validated
iqk kquant kernels and corrupted Hy3 output. It was **caught on live output at 20:47Z,
not by tests** (fixed same day by `1977a5d78` plus static_asserts).

So: a patch that changes an activation quant type, a type-traits row, or a dispatch
table entry is in a known-dangerous class whose failure mode the gate cannot see. Say
so in the hypothesis, and prefer a mechanism that leaves type selection alone.

## No keep or fold may drop a kernel path (kernel_coverage.py, 2026-10-04)

`INC-20260925-parallel-repack-lost-in-bundled-revert`: the OpenMP repack
parallelisation (`52ddd3200`) rode in bundle `814e81782` beside two failing changes.
Reverting the bundle (`358f0c748`) silently removed it. v7-v10 then loaded every model
single-threaded until 2026-09-25, and no serving A/B could see it. The class
before that was `INC-20260706-iqk-missing-subsystem`. Every keep now runs a
preservation gate after its A/B and before the champion moves (~1 s, artefacts the
loop already has). A loss is a HARD veto (`KEEP_CANDIDATE-kernel-coverage`):

- **static**: `nm` families of the anchor vs candidate DSOs. These are repack
  `tensor_traits<block,INTER,COLS>`, `ggml_gemm_*`/`ggml_gemv_*`, OpenMP outlined
  regions (`[clone ._omp_fn.N]`, how the lost repack shows), tinyBLAS gemm
  instantiations, `iqk_*` and `ggml_compute_forward_*`. Counted.
- **source**: a `git grep` inventory. This covers `#pragma omp` and `GGML_IQK` per
  file, repack trait definitions, gemm/gemv definitions, `getenv` knobs, CMake
  `option()`s and `tests/` files.
- **runtime**: what EXECUTED for this target's serving shape, from the A/B launches'
  stderr. These are `[iqk] ACTIVE:` per quant type (dense and MoE), extra-buffer
  sizes (`CPU_REPACK` shrinking means tensors lost their repack) and per-tensor
  repack lines. Markers the anchor's own launches disagree on are never failed on.

If a patch deliberately replaces a path, say so in the hypothesis statement or as a
comment on an added line: `KERNEL-REPLACES: <old key glob> => <new key glob>`, for
example `KERNEL-REPLACES: omp_region:ggml_repack_row_groups* => omp_region:my_pool`.
The new key must exist in the candidate. The replacement must also be measured: a
runtime loss on model T needs an A/B on T, and a static or source loss needs an A/B on
every bound target. Folds and forward-ports run the same check without a loop:
`python3 -m scripts.kernel_rnd.autokernel.loop.kernel_coverage fold-check --repo <tree>
--base <champion> --candidate <ref> [--base-build B --candidate-build C]`, which
`fold2_gates.py` runs as G0. Declarations there come from commit messages.

## ONE champion; lanes keep on working branches and fold into it (cross_target.py, 2026-10-04)

A shared change can help one model and hurt another: DS41's held keeps `b3e0b0902`
cost Qwen3.8-Flash-Next 29-41% of prefill. There is still exactly ONE champion per
production kernel tree (`champion.CANONICAL_BRANCH`). Production is one CPU build and
one GPU build of that one commit. Never fork a second champion or trunk.

With a lane binding (`lane_targets.py`), each target keeps on its own WORKING branch
(its `--experimental-branch`), so its anchor and accumulator stay as they are. A keep
that regresses no peer folds into the champion straight away. "Regresses no peer"
means every peer's A/B is within its floor or bar, and every peer's executed-kernel
manifest is unchanged; a changed manifest is allowed only with a decisive peer gain.
The source coverage gate G0 runs first. Other lanes pick the change up from the
champion at their next keep.

A keep that regresses a peer is still KEPT on your working branch, as
`target_only_pending_gate`. It cannot reach the champion, or production, until it is
gated. The loop then puts one queued hypothesis in your inbox,
`00-gate-<patch>-aN.md`. When you see one, propose a patch that keeps the mechanism
for your target only, through runtime dispatch, so every other model executes exactly
the champion's kernel path.

**The gate must key on something runtime_attestation can see.** That means GGUF
metadata (`general.architecture` / `LLM_ARCH_*`, hparams, tensor shapes, quant types)
or llama-server argv (cparams). An environment-variable gate (`getenv`) is vetoed
(`KEEP_CANDIDATE-gate-attestation`).

Put `GATES-KEEP: <patch id>` in the hypothesis statement. The other lanes then each
fold-check the whole held series on their own target, and when all of them approve,
the series folds into the champion. If a lane rejects it, the next gate attempt is
queued, up to 3 attempts. After that the keep stays on your branch, out of production,
and the ledger records it as `gate_exhausted`.

Every decision is a line in `<binding dir>/cross-target-ledger.jsonl` and rides in the
keep's experiments-DB row. So a peer regression is no reason to abandon a mechanism
that wins on your target; gate it.

## Not this loop's surface — do not propose these

The backlog is full of measured levers that are NOT kernel-source patches. They are
real work; they are simply not what this loop can build, gate and A/B:

- **Serving and sampling flags** — spec-dec draft depth, `--spec-type ngram-*`,
  `-ctk/-ctv` KV quant, batch/ubatch geometry. Config, not source.
- **Drafter work** — MTP/EAGLE3 heads, tree-draft, drafter fine-tuning for acceptance
  rate. A separate track with its own handoffs.
- **Lossy model surgery** — vocabulary pruning, layer/depth removal. Out of scope by
  the ratified lossless-only rule, and both contributed to a 15/40-point degradation.
- **KT / trellis quants** — CPU-only, so the MI210 cannot participate at all. Also
  dominated on the arithmetic that motivated them: `block_iq2_kt` is 68 B per 256
  weights = **2.125 bpw** against **IQ2_XXS at 2.0625 bpw**, so it is *larger* than
  what GLM-5.2 already uses and saves zero bandwidth while adding per-weight work. A
  quality-at-equal-bpw play mis-framed as a speed play.

## Settled — do not re-open without new evidence

- **`MMQ_MFMA` OFF.** +23.09% on the 0.5B toy, **+0.50%** on Qwen3.8-27B. Real where
  it was taken, worth nothing where the fleet runs.
- **`ubatch 512→1024`.** A NULL ARM: llama.cpp clamps `n_ubatch = min(n_batch,
  n_ubatch)`, so both arms ran at 512 on one identical binary. The +46.9% was a
  bimodal sample whose median landed on the fast mode.
- **ngram 2.8×.** Retracted — a warm-context self-copy artifact; −17.4% on 122B-IQ2.
- **Already in v9:** `GGML_IQK` (since v8), MMQ `a6b4b5263`, HIP graphs (upstream
  default ON).

## Authority

This package **screens**. It never promotes. Every result is non-promotable by
construction, and promotion is `docs/reference/kernel-freeze-runbook.md` — seven
steps, ~100 lines, and it shipped v7, v8 and v9.

Under `P-AK-SEARCH-1` denial 4, prior records inform **hypothesis formation** only;
`experiments.ExperimentStore.recall(ranking_authorized=False)` is the default until
the operator amends it (decision D1,
`handoffs/active/autokernel-rebuild-program.md`). Cross-epoch records are returned but
marked `stale_epoch` — the fact that a mechanism was tried is formation; its *number*
is not comparable. `P-AK-SEARCH-1-A3` permits epoch-scoped ranking
(`--rank-prior-experiments`), and its do-not-repeat gate refuses re-proposing a
mechanism already answered three times in the same epoch.

**The epoch is the MEASUREMENT epoch** (OP-60, 2026-09-26): the anchor commit, the
build recipe and the declared host state (execution and request digests, target,
screen scope, serving instrument), **excluding actor configuration** — the planner,
critic and author models and their fallbacks. Changing an actor does not make prior
same-anchor results stale and does not reopen a mechanism the do-not-repeat gate has
closed; changing the anchor, recipe or host state does. A record whose measurement
identity cannot be established is compared on its full epoch only.
