# C6 evaluator policy

This is the hardware-independent correctness and admission policy used by
`c6_reward_integrity.py`. It does not authorize a production decision and it
does not run a semantic judge, GPU benchmark, or retrieval system.

## Pinned numerical source

The adopted numerical baseline is `flashinfer-ai/flashinfer-bench` commit
`40e6ca7844b514eb4b1c7edba6d6a7377df57870`, read without executing the
external repository:

- `flashinfer_bench/bench/config.py`: default `atol=1e-2`, `rtol=1e-2`, and
  three numerical trials in the upstream benchmark configuration.
- `flashinfer_bench/bench/utils.py`: an element is outside tolerance only when
  `abs_error > atol` **and** `rel_error > rtol`; matched ratio is one minus the
  fraction of those elements.
- `flashinfer_bench/bench/evaluators/lowbit.py`: default low-bit required
  matched ratio is `0.95`.
- `flashinfer_bench/bench/evaluators/default.py` and `lowbit.py`: output dtype
  mismatch and non-finite output refuse correctness, and maximum absolute and
  relative errors are retained.

Our contract may tighten the two per-element bounds and the matched ratio. It
cannot loosen them. Required output dtype and accumulator dtype are
operator-owned structural fields and are checked before values are compared.
The accumulator evidence must come from parent-controlled static/IR inspection
and is content-hashed; a candidate self-report is not evidence.

Three repeated executions are required and compared bitwise. Three is exact,
not a minimum: an implementation cannot stop on the first matching pair or the
first mismatch. Every execution receives a distinct clone of one pristine
input snapshot, and the reference retains an independent pristine input; a
candidate cannot move the oracle by mutating shared inputs. Exception-handler
fallback paths are mutation-tested by replacing each entire handler with a
hard raise and re-running the wrapper, including assignment-plus-fall-through
returns. This records the ninth vacuous-verification shape, **wrapper launders
a kernel failure into the
reference**.

The fixed mutant driver accepts accumulator evidence only when the complete
source digest and normalized kernel-function AST digest match its trusted
allowlist. The observed accumulator dtype comes from that allowlist, not from
the task's required dtype field. Any source or function drift refuses before
the numerical gate.

## Gate topology

The retained tiers are `L1_static`, `L2_ghost_replay`, and `semantic_judge`.
The old `L3` tier is dropped. The semantic judge is observable but non-gating
until one calibration record rejects all three fixed `c6_mutants`:

- `layernorm_no_affine`
- `softmax_no_maxsub`
- `matmul_transpose_no_t`

No missing, extra, or partially rejected corpus can enable it. Unknown GPU
parts refuse; the policy has no generic roofline or hardware fallback.

## Write-side admission

`epyc.autokernel.c6_admission_receipt.v1` retains the raw first-turn and
verification latency pairs and recomputes:

`r_verify >= max(beta, alpha * r_1)`, with `alpha >= 1.2` and `beta >= 1.2`.

Every call must supply a predeclared finite implausible-speedup cap. There is no
global guessed cap. A ratio above the supplied cap is persisted as
`implausible_speedup_refused`, never admitted. Candidate, anchor, and evaluator
commits are exact 40-hex bindings; both the first turn and verification re-run
must carry exact successful correctness verdicts; `reopen_when` is mandatory.
The receipt and the prospective Vidya capture are independently self-hashed.
The append-only store writes all outcomes and implements no read/retrieval
memory.

The matching root-side adapter-table row still requires application by the
owning root session. Proposed row:

> AutoKernel C6 admission receipts (`epyc.autokernel.c6_admission_receipt.v1`)
> | measurement | candidate — producer emits the identity-bound
> `epyc.vidya.autokernel_c6_admission_capture.v1`; adapter must validate both
> self-hashes, commit/reopen bindings, raw latency-derived ratios, threshold and
> implausible-cap disposition, then project through the existing `ClaimTuple`
> ladder without defining a new grade | —

## Separable records

The G15 retrodiction is pure and hardware-free: on frozen-v9 B=128 shares it
must select gather/scatter (18.631%) over recurrent (17.464%) and
norm+activation+elementwise (1.490%). The per-round reflexion record carries
diagnosis/fix verdicts, expected/actual outcomes, estimated/achieved speedup,
estimate error, and lesson/avoid/try lists. It is a per-round write record only;
cross-run memory remains out of scope.
