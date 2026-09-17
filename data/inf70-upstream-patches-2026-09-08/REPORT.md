# UP-1 / UP-2 -- four upstream ggml/llama.cpp contributions, prepared for operator review

Read-only with respect to /workspace and the production tree. No inference, no
benchmarks, no bench-region lock. All verification below was done against a fresh
git clone of https://github.com/ggml-org/llama.cpp (upstream master,
e71b805 -- Revert "CUDA: size routed MoE MMQ N-tiles ... (#24546)" (#28551),
ggml revision e71b805, ggml version 0.23.0, cloned/verified 2026-09-07), never against
our vendored/frozen tree and never against a README. No issue or PR was opened, and
nothing was pushed anywhere. Everything below is prepared text and patches for the
operator to submit at their discretion.

Work directory: /mnt/raid0/llm/tmp/inf70/agents/up/. Upstream clone (scratch, not
recorded in this repo's own history, safe to delete): /mnt/raid0/llm/tmp/inf70/agents/up/upstream/
-- currently on master, working tree clean, no local branches or stashes left behind.

## Status at a glance

| # | Contribution | Confirmed present on current upstream? | Patch | Reproducer/evidence |
|---|---|---|---|---|
| 1 | Vectorise SIGMOID (ggml_vec_sigmoid_f32 dead code) | YES -- vec.h:936 was still the scalar loop, unary-ops.cpp still routes f32 sigmoid through per-element expf | 01-sigmoid-vectorize/0001-vectorise-sigmoid.patch | acc_check.c (accuracy vs libm), test-backend-ops |
| 2 | (row, column-chunk) split for elementwise ops at batch 1 | YES -- get_thread_range() in common.h still row-only; ggml_compute_forward_repeat_f32/_f16 still `if (ith != 0) return;` | 02-rowcol-split/0001-rowcol-split-elementwise.patch | bench_rowcol.c (timing), test-backend-ops |
| 3 | SET_ROWS wdata heap overflow | YES -- planner still sizes by n_tasks (fixed at 1), kernel still splits over n_threads | 03-set-rows-overflow/0001-set-rows-wdata-overflow.patch | repro_set_rows_overflow.c, ASan-confirmed |
| 4 | MAP_CUSTOM*/CUSTOM ignore requested n_tasks | YES -- forwards in ops.cpp still pass params->ith/params->nth straight through | 04-map-custom-ntasks/0001-map-custom-n-tasks-contract.patch | repro_map_custom_ntasks.c + regression_check.c |

All four defects are real, all four are still present on current upstream, and none
were found to already be fixed. All four patches apply cleanly with `git apply --check`
against a pristine master checkout, independently of each other (verified together in
one working tree at the end of this session -- see Section 6).

## How each was verified (methodology, applies to all four)

1. Read the actual upstream source, not the vendored copy in /mnt/raid0/llm/llama.cpp
   (frozen production, production-consolidated-v9, a much older and independently
   patched tree) and not any README or handoff doc. Every file/line cited below is from
   the scratch clone.
2. Built ggml standalone (`cmake -S . -B build ... -DGGML_NATIVE=OFF
   -DBUILD_SHARED_LIBS=OFF`, target `ggml` or `test-backend-ops`) rather than the whole
   llama.cpp toolchain, to keep iteration fast; test-backend-ops needed
   -DLLAMA_BUILD_TESTS=ON (it links libllama/libllama-common).
3. For the two overflow/contract bugs (#3, #4): wrote a minimal, self-contained
   reproducer using only the public ggml.h/ggml-cpu.h API, built it with
   -fsanitize=address,undefined, ran it against pristine upstream to reproduce the
   defect, then against the same tree with the one patch applied to confirm it's gone,
   with no other change in between. No destination-aliasing write-primitive or
   step-by-step extraction path was written -- per the task constraint, these are
   presented as a defect class plus a fix, not exploits.
4. For the two performance contributions (#1, #2): re-ran the correctness suite
   (test-backend-ops -b CPU, explicitly noting that flag is required -- the harness is
   vacuous on this backend without it) and, for #2, an isolated timing check against
   real upstream code (not our fork's numbers) to give the PR real, freshly-measured
   figures rather than only re-quoting a downstream measurement.
5. Every patch was generated as a `git diff` against the exact upstream revision named
   above, with no campaign-specific env-gate, naming, or attribution knob left in it
   (see Section 5, "what was deliberately stripped").

## Section 1 -- Sigmoid vectorisation (UP-1a)

Confirmed present on current upstream. ggml/src/ggml-cpu/vec.h:936 is still:
```
inline static void ggml_vec_sigmoid_f32 (const int n, float * y, const float * x) { for (int i = 0; i < n; ++i) y[i] = 1.f / (1.f + expf(-x[i])); }
```
with zero callers, and ggml_compute_forward_sigmoid (unary-ops.cpp) still routes
through unary_op<op_sigmoid> -- one expf() libm call per element, on every thread's
row, regardless of shape.

Fix: add ggml_v_sigmoid next to every existing ggml_v_silu (AVX-512, AVX2, SSE2,
SVE, NEON, RVV) -- same ggml_v_expf kernel, numerator 1 instead of x -- make
ggml_vec_sigmoid_f32 a real SIMD kernel in vec.cpp, and add a
ggml_compute_forward_sigmoid_f32 in unary-ops.cpp (same row-split shape as the
existing ggml_compute_forward_silu_f32 in ops.cpp) for the f32->f32 case; every other
type combination is untouched.

Framing, stated honestly, per the operator's instruction: this is presented to
upstream as a straightforward missing-vectorisation fix, unconditional, with the same
precedent as silu. It is NOT presented as a win our own stack banked -- on our fork
it measured +0.0 pp once our elementwise ops were already thread-split at batch 1 (the
vectorisation and the thread-split fix the same bottleneck from two different angles
once the split is in place -- the split alone captured the win), and it cost a
measured top-1 token flip at ~19 generated tokens in one of three greedy test prompts,
from the ulp-level numeric change. We did not adopt it on our own stack. Its value is
upstream-only, for any CPU build/config that does not already carry an equivalent
batch-1 thread split -- see contribution #2 below, which is exactly that split, offered
as its own, independent PR. The PR body ("Note for reviewers" in
01-sigmoid-vectorize/PR_BODY.md) states this explicitly rather than presenting the
change as a measured win.

Accuracy, re-measured here (not re-quoted from the campaign) against current
upstream, 4,194,304 inputs over +-40 plus edge cases:

| metric | measured here (this session, upstream e71b805) | previously reported (downstream fork, different compiler/libc) |
|---|---:|---:|
| bitwise exact vs libm | 60.29% | 74.2% |
| within 1 ulp | 94.29% | 92.1% |
| max abs error | 9.207e-08 | 1.19e-07 |
| max rel error | 2.554e-07 (x~=-16.78) | 2.75e-07 (x~=-6.38) |
| out of [0,1] | 0 | 0 |

The two measurements disagree on the exact percentages (expected -- they depend on the
host's libm/compiler) but agree qualitatively: no bitwise identity, same order of
magnitude error, output always in range, exact at the special values. Both numbers
are shown in the PR body labelled by source, rather than presenting either as
definitive.

Testing: test-backend-ops -b CPU -o SIGMOID: 8/8 passed. Broader sweep
(SIGMOID,SILU,ABS,NEG,SGN,STEP,TANH,ELU,RELU,HARDSIGMOID,HARDSWISH,EXP): 96/96 passed.
-b CPU used explicitly both times -- without it this harness is vacuous on this
backend.

Directory: 01-sigmoid-vectorize/ -- patch, PR body, accuracy-check source (acc_check.c)
and its output, build/test logs, and the original downstream diff for reference
(campaign-commit-086b5b9c9.diff, not part of the patch -- kept only as provenance).

## Section 2 -- (row, column-chunk) split for elementwise ops at batch 1 (UP-1b)

Confirmed present on current upstream. get_thread_range() (ggml/src/ggml-cpu/common.h)
is still a pure row split; ggml_compute_forward_repeat_f32 and the newer
ggml_compute_forward_repeat_f16 (this second function did not exist in the
downstream fork's base revision -- upstream has grown a variant since; both were
patched here) are still `if (params->ith != 0) return;` outright.

Fix: get_rowcol_split() (new, common.h) enumerates (row, column-chunk) tasks
instead of rows alone, degenerating to the exact old behaviour when there are already
enough rows to occupy every thread or the row is too short to be worth chunking (512
elements, ggml_rowcol_min_elems()). Applied to apply_unary_op, apply_unary_op_functor
(unary-ops.cpp), apply_binary_op (binary-ops.cpp), and both
ggml_compute_forward_repeat_f32/_f16 (ops.cpp). Bit-identical by construction -- every
output element is still produced by exactly one thread from the same input bytes.

Shipped unconditionally, no env gate -- unlike the sigmoid PR this one changes no
numerics at all, so there is no reason to gate it. (The downstream fork gated it
GGML_ROWCOL_SPLIT/GGML_ROWCOL_MIN_ELEMS, default off, purely so that fork could
attribute this lever separately from others accumulating on the same tree; that
attribution reason does not apply to a standalone upstream PR, so both env lookups were
removed and the 512-element threshold is now a plain constant.)

Performance, re-measured here against pristine upstream (not re-quoted from the
downstream fork's serving numbers, which were measured on top of unrelated stacked
levers): a standalone SIGMOID([10240,1,1,1]) micro-benchmark (bench_rowcol.c, no
sigmoid vectorisation applied -- isolates this patch's effect alone):

| n_threads | before | after | delta |
|---:|---:|---:|---:|
| 1 | 14.735 us | 15.130 us | +2.7% (split doesn't fire at nth=1, within noise) |
| 8 | 22.640 us | 11.118 us | -50.9% |
| 48 | 30.708 us | 17.723 us | -42.3% |
| 96 | 48.616 us | 29.878 us | -38.5% |

Testing: test-backend-ops -b CPU across every touched op family plus REPEAT
(29 ops): 424/424 passed, 0 failures.

A caveat for our own campaign's accounting, not for the PR -- surfaced by SYNC-17
this session, after this contribution was originally drafted: on our own experimental
fusion tree (not upstream), a separate mechanism called TINY_SOLO re-dispatches
certain barrier-elided nodes with sp.nth forced to 1, which means this split's own
firing condition (nth > 1 && nr < nth) never triggers on those nodes in our stack's
512-4096-element band -- TINY_SOLO and this split are two of our own barrier/occupancy
mechanisms stepping on each other on our fork. This has no bearing on the upstream
patch: pristine upstream ggml has no TINY_SOLO, so the split fires unconditionally
there whenever nr < nth and the row is long enough, exactly as measured above. It is
noted here only so the campaign's own internal accounting for "S" (the split, in prior
write-ups) doesn't double-count nodes that TINY_SOLO already touches on our fork; it
does not change the correctness, applicability, or the measured numbers of the patch
itself.

Directory: 02-rowcol-split/ -- patch, PR body, standalone benchmark (bench_rowcol.c)
and its logs, build/test logs, and the original downstream diff for reference
(campaign-commit-2af8669ce.diff, kept only as provenance -- the actual patch differs
from it: no env gate, and the ggml_compute_forward_repeat_f16 half was added new here
since it didn't exist when the downstream diff was authored).

## Section 3 -- SET_ROWS wdata heap overflow (UP-2a)

Confirmed present on current upstream, verified against the real source, not a
README. ggml_graph_plan() (ggml-cpu.c) still sizes the SET_ROWS scratch buffer
as `... * n_tasks` with n_tasks fixed at 1 for this op; ggml_compute_forward_set_rows_impl
(ops.cpp) still row-splits unconditionally over params->nth == n_threads and
indexes a private wdata slice per thread on the F16-source -> non-F16-destination
branch.

One correction to how this was described in an earlier internal write-up, found
while verifying against the real upstream source rather than trusting the prior
report: that write-up quoted a comment at a specific line claiming the CPU backend
documents SET_ROWS's single-task scheduling as a guard against a "duplicate
destination index" race. No such comment exists in real upstream -- it exists only
in our own experimental fork, added by a prior session as that fork's own explanatory
comment, not inherited from upstream. What upstream does document, correctly, is at
the ggml_set_rows() builder API level (ggml.h): "undefined behavior if destination
rows overlap" -- a real, pre-existing, already-documented precondition, not a new
finding. This PR does not claim a race-condition finding; it claims exactly one thing,
the heap overflow, described below.

Reproduced with ASan on pristine upstream master (e71b805): a minimal one-node
SET_ROWS graph (F16 source, F32 destination, 8 rows, 8 threads, no destination
aliasing) plans a 768-byte buffer while the kernel needs >=2560 bytes; running it
reports heap-buffer-overflow inside ggml_compute_forward_set_rows_impl ->
ggml_fp16_to_fp32_row. With the one-line fix (size by n_threads instead of the
advisory n_tasks), the same reproducer plans exactly 2560 bytes and exits clean.
test-backend-ops -b CPU -o SET_ROWS -p "type_src=f16" (280 cases) still passes with
the fix; it was also confirmed not to already catch this on pristine upstream --
its largest nr in that filter is 11, and hardware_concurrency() on the build machine
was 192, so none of its shapes reach the nr >= n_threads condition that triggers the
overflow.

No exploit or extraction path is included -- a counting/ASan-detection reproducer only,
per the task's hard constraint on the two overflow bugs.

Directory: 03-set-rows-overflow/ -- patch, PR body, reproducer source and logs
(before/after ASan runs), test-backend-ops regression logs.

## Section 4 -- MAP_CUSTOM1/2/3/CUSTOM ignore the caller's n_tasks (UP-2b)

Confirmed present on current upstream. ggml_get_n_tasks() (ggml-cpu.c) still
honours the caller-registered n_tasks (including the GGML_N_TASKS_MAX sentinel)
when planning; ggml_compute_forward_map_custom1/2/3 and ggml_compute_forward_custom
(ops.cpp) still pass params->ith/params->nth (the graph's real thread indices)
straight to the user callback, never the planned n_tasks.

Background, stated correctly per the operator's framing: ggml_get_n_tasks() is
advisory on this backend precisely because ggml_graph_compute_thread() overwrites
params.nth to n_threads for every node before dispatch -- the planner's per-op
decision never reaches the kernel unless the kernel itself re-derives it. Every other
CPU kernel is written to not care (it just uses params->nth as "how many threads
exist," which is always true). MAP_CUSTOM*/CUSTOM are the one place this matters
differently, because the caller, not ggml, chose n_tasks and was promised (by the
documented public API) that the callback would see it.

This is a contract violation, not (by itself) a memory-unsafety bug inside ggml --
it becomes one in third-party code that follows the documented contract (e.g. sizing
per-thread scratch by the n_tasks it registered with). Framed that way in the PR
body, with the class of problem named and no assumed victim code written.

Reproduced: a MAP_CUSTOM1 op registered with n_tasks = 1 (single-threaded request)
is invoked 8 times with nth = 8 on each, at n_threads = 8, on pristine upstream
master. Fix: clamp the effective n_tasks in each of the four forwards (mirroring the
planner's own GGML_N_TASKS_MAX logic) and early-return on threads beyond it. With the
fix, the same reproducer shows exactly 1 invocation with nth = 1. A second, separate
check confirms the common GGML_N_TASKS_MAX (full-parallelism) case is unaffected:
8/8 threads still run.

No existing test coverage. `grep -rl map_custom tests/` on upstream master returns
nothing -- this contract has zero regression coverage today; the two small programs in
this directory are the only coverage that exists for it, before or after the fix.

Directory: 04-map-custom-ntasks/ -- patch, PR body, reproducer + regression-check
sources and logs.

## Section 5 -- What was deliberately stripped from the downstream diffs

None of the four patches carry the downstream fork's own naming, attribution, or
knobs:

- No GGML_VEC_SIGMOID / GGML_ROWCOL_SPLIT / GGML_ROWCOL_MIN_ELEMS environment
  variables -- both performance patches are shipped as unconditional fast paths
  (sigmoid follows the existing silu precedent for a change that is not bit-identical;
  the row/col split has no numerical change at all, so gating it would have no purpose
  upstream).
- No INF-70/SYNC-n campaign references in code comments -- replaced with plain,
  self-contained explanations of the mechanism.
- No campaign-specific message material (measured-on-our-fork numbers, D9-ack
  lines, hook-defect notes) carried into the patch or PR body; where a downstream
  number is cited at all (sigmoid's earlier accuracy measurement, the top-1 flip), it
  is explicitly labelled as a downstream/fork measurement, separate from what was
  re-verified here on real upstream.

## Section 6 -- Final state

- Scratch upstream clone: master checked out, working tree clean, no local branches,
  no stashes. (Three throwaway local branches -- up-set-rows-fix, up-rowcol-split,
  up-sigmoid-vectorize -- were created and used only to isolate builds during
  verification; all were deleted, their diffs live only as the four .patch files.)
- All four patches independently re-verified with `git apply --check` against the same
  pristine master tip in the same final pass (see "How each was verified", step 5).
- Large build directories were removed after each contribution's checks completed;
  only source files, patches, PR bodies, and text logs remain under this work
  directory. The compiled ASan reproducer binaries (repro*, regression-check,
  acc_check) are kept for the operator to re-run without rebuilding, and can be
  deleted freely -- they rebuild in well under a minute each from the .c sources with
  the command in each file's header comment.
- No process was left running; no PID was killed by name pattern (none needed to be --
  every reproducer here runs to completion and exits on its own); no bench-region lock
  was taken; /mnt/raid0/llm/llama.cpp (frozen production) was never read from for code
  (only cited by name above, for contrast) and never written to.

## Section 7 -- Out-of-scope item noticed in passing, not actioned

The handoff's WRAP-5 row ("report the llama.cpp GGUF C reader segfault upstream --
fold into UP-2") is unowned and was not part of this task's explicit four-item scope;
it was not investigated, reproduced, or written up here. Flagging its existence only
so it isn't mistaken for covered -- it is not one of the four items in "Status at a
glance" above and needs its own dispatch.
