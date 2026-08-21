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

The value tier now imports the ratified policy in
[`../C6_EVALUATOR_POLICY.md`](../C6_EVALUATOR_POLICY.md): required output and accumulator dtype
precede the pinned FlashInfer-Bench elementwise/matched-ratio gate, NaN/Inf refuse, max errors are
retained, and every candidate arm runs exactly three times with bitwise comparison. The retained
topology is L1 + L2 + semantic judge; L3 is dropped. Optional `--semantic-judge-verdicts FILE`
consumes (but never produces) the fixed three-mutant calibration map. The judge remains non-gating
unless all three values are `REJECT`.

The vacuous-verification catalogue also has a ninth shape: **wrapper launders a kernel failure into
the reference**. `c6_reward_integrity.replace_fallback_returns_with_raise` mutates exception-handler
fallback returns into a hard raise and requires a trusted re-run; the CPU adversarial tests prove a
laundered wrapper is exposed. This wrapper-level mutant is kept outside the three numerical omission
tasks so it cannot silently change the semantic-judge calibration corpus.

## Status 2026-08-21
- L1 static arm: **all six candidates PASS** (0 findings) — scanner validated non-vacuous first.
- GPU arms (L2 ghost replay + value oracle, standard + adversarial): **pending a negotiated window**.
- Downstream: the semantic-judge tier (RVP-C6-19, operator-ratified) must reject all three mutants
  before it gates anything — this corpus is its minimal validation set. Until then it is explicitly
  non-gating.

MEASUREMENT.md: observations only; nothing here gates a keep/revert/deploy/promote decision.
