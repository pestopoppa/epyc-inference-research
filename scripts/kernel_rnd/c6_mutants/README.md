# RVP-C6-20 — the omission-mutant falsification harness

**Claim under test:** "L1 (AST blacklist) + L2 (Ghost Replay) is a sufficient C6 gate."
**Method:** three hand-authored gfx90a Triton kernels, each a GENUINE kernel doing real `tl.*`
work that computes LESS than its operator — TritonRL's rule-3 OMISSION class
(`intake-1241#record`, arXiv:2510.17891v2 Appendix H), which is invisible to a symbol scan
(nothing forbidden appears) and to Ghost Replay (the kernel IS load-bearing).

| Task | Omitted | Why it passes at standard inputs |
|---|---|---|
| `layernorm_no_affine` | scale (γ) + bias (β) | default module init (γ=1, β=0) makes the omission an **exact identity** |
| `softmax_no_maxsub` | max-subtraction | mathematically identical; only large-\|x\| inputs overflow |
| `matmul_transpose_no_t` | trailing transpose | square shapes defeat the shape check; **predicted value-visible** at random inputs — included to measure where the omission class splits |

## Files
- `mutants.py` — 3 × {mutant, honest positive-control, PyTorch reference, standard+adversarial inputs}
- `l1_scan.py` — L1 as specified (KernelGenBench blacklist + TritonRL rule 1); static, CPU
- `test_static.py` — **mutation-tests the scanner itself**: 8 planted-dirty samples it must flag,
  a scope negative-control (the references must FAIL), and an empty-scope refusal
- `run_falsification.py` — driver; one JSONL row per (task, candidate, tier, arm); asserts the
  row count and the honest-arm positive controls before concluding anything

## Running
```bash
python3 test_static.py                 # CPU: validate the scanner (6 tests)
python3 run_falsification.py           # CPU: L1 arm only
python3 run_falsification.py --gpu --i-have-a-window --out results.jsonl   # MI210, ~3 min
```
`--gpu` REFUSES without `--i-have-a-window` (negotiate with the parallel agents; shared MI210),
refuses on a non-gfx90a device (never estimate an unknown part), and refuses to conclude on a
partial row count or a broken honest arm.

## Status 2026-08-21
- L1 static arm: **all six candidates PASS** (0 findings) — scanner validated non-vacuous first.
- GPU arms (L2 ghost replay + value oracle, standard + adversarial): **pending a negotiated window**.
- Downstream: the semantic-judge tier (RVP-C6-19, operator-ratified) must reject all three mutants
  before it gates anything — this corpus is its minimal validation set.

MEASUREMENT.md: observations only; nothing here gates a keep/revert/deploy/promote decision.

## RVP-C6-15 — LIBRARY-SUBSTITUTION static hostile family

`library_substitution.py` adds three owned candidate sources for Torch matmul,
rocBLAS GEMM and hipBLASLt matmul. The native entry is each source's
`candidate(...)`; `scan_source(..., candidate_functions=['candidate'])` follows
same-module helpers and import/from-import, module and local callable aliases.
Known Torch targets are matmul, mm, bmm, addmm, einsum, softmax, layer_norm,
compile, `torch.ops.aten.*` and `torch.nn.functional.*`. Known vendor namespaces
are rocblas, hipblas and hipblaslt, with GEMM/GEMV/matmul/TRSM/SYRK call names.
This is a finite AST detector, not a whole-program dispatch proof.

Allocation, `.to`/`.contiguous`, dtype metadata, and reference/validation code
outside the candidate's reachable functions remain allowed. Dynamic getattr,
callback parameters, indexed dispatch, conflicting module/local conditional
aliases, and ambiguous or nested helper definitions are UNKNOWN rather than a
clean result. Arbitrary native extensions, external helper
implementations, reflective mutation, and general Python control/data flow are
outside this detector's coverage; it cannot prove their absence.

The existing driver emits four separate `L1_library` rows (three hostile and one
honest). Its summary requires every hostile case to FAIL with a nonzero library
finding count, validates the exact case identity set, and reports the flagged
case and finding totals. Missing, duplicate, clean-verdict or zero-finding
mutations refuse a conclusion. The original six static / 25 full-stack row
denominators and the three GPU task predicates are unchanged.

The model-free tests execute only these fixed sources through restricted fake
imports and in-memory provider-call traces, then check the native AST screen and
reporter. Stub calls establish fixture delegation only; they are not HIP kernels,
GPU dispatch observations, speedups, correctness-oracle outcomes or evidence that
live AutoKernel T0 admission is complete. No GPU is needed for this family.
