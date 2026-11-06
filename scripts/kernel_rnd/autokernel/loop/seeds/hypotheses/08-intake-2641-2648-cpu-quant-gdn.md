# CPU lane: intake 2641-2648 quantization and GDN seeds (DS41 / Q38FN)

Source: epyc-root `handoffs/active/quantization-and-low-bit.md` (QUANT-HEAD-1, QUANT-ROT-1) and
`handoffs/active/model-architecture-readiness.md` (GDN-SHARE-1, GDN-ORACLE-1), filed by root commit
c5339814a. Operator 2026-11-06: "Take all four but turn the quantization and kernel experiments into
detailed autokernel hypotheses". Owner of all four: AutoKernel.

Champion anchor: C7r = llama.cpp `62cb6a6af` (CAS `a6a560d9d`). Lane: CPU. Campaigns: DS41 and Q38FN
(both CPU-served in production). Grade on arrival: `design_prior` -- nothing below is measured.
Gate class for every seed: **futility-only** (interim looks may stop for futility, never declare a
win, until OP-ABSEQ / OP-PQP / OP-ALPHAWIN are decided). No effect size below is measured; every
estimate is marked [inferred] with its arithmetic and symbolic where a model dimension is not yet audited.
`change_class` values are from the closed set in `hypothesis_portfolio.SOURCE_MANIFEST_CHANGE_CLASSES`.

Ordering (prerequisites):

    AK-GDN-S1 (share probe, CPU) --core share < 5% both phases--> close LinQuant line (oracle keeps correctness value only)
                                 --core share >= 5% either phase--> AK-GDN-O1 (oracle) --> any GDN kernel seed (none filed yet)
    AK-QH-A (audit) --> AK-QH-B (one Q4_K head arm, per model)
    AK-QR-1 (offline, no kernel) --> only a positive arm may seed a converter/kernel hypothesis later

## AK-GDN-S1 -- GDN recurrent-core share of CPU decode and prefill time (measurement seed, runs first)

- **Target:** DS41 and Q38FN, CPU lane, C7r; op `GGML_OP_GATED_DELTA_NET` (chunked GDN graph: WY prep +
  state pass), separated from projections, attention and MLP.
- **change_class:** none (measurement only; no source change; not authorable).
- **Metric (shared definition with the MI210 half, main-inf22):** `gdn_core_share[phase] =
  sum(time in the GDN recurrence-core ops) / total step wall time`, reported separately for
  phase = decode (tg, per token) and phase = prefill (pp, per ubatch). Method on CPU = the loop's
  existing CPU profile path (`loop/cpu_profile.py`, perf sampling attributed to symbols), at the codified
  DS41 / Q38FN serving recipe and shape. The MI210 half must use the same numerator/denominator per
  phase (rocprofv3 kernel time) so the halves compare; if main-inf22 has published a different
  definition, this one is the CPU statement and the difference must be reconciled before closing.
- **Expected effect:** none (it decides routing). Prior [inferred]: GDN layers in a hybrid stack are a
  minority of per-token bytes, so decode share is likely small; prefill share is the open question.
- **Stop-on-answer:** both phases measured on both models with two valid samples each, or the first
  model whose two phases both clear 5% (line stays open).
- **Stop-on-futility / falsifier (pre-registered):** core share < 5% in BOTH phases on BOTH campaigns ->
  close the LinQuant implementation line as no-transfer; AK-GDN-O1 then lands for correctness value only.
- **Correctness gate:** none (no code change). Estimate: ~2 CPU-hours (handoff box).
- **Citation:** intake-2641#record, intake-2642#record.

## AK-GDN-O1 -- fp64 CPU conformance oracle for the chunked GDN graph (enabling prerequisite)

- **Prerequisite:** AK-GDN-S1 answered. Lands BESIDE the existing AKX-ALL-31.GDN one-hot oracle
  (`loop/gdn_reference.py` + `loop/gdn_reference_probe.cpp`, and its `test-backend-ops` case on the
  champion-derived experimental branch), as a second case in the same harness. Not a new harness.
- **Target:** CPU lane, C7r-derived `llama.cpp-experimental` branch; op `GGML_OP_GATED_DELTA_NET`
  (chunked path and per-token path); test file beside `tests/test-backend-ops.cpp`.
- **change_class:** `oracle_port`.
- **Mechanism:** vendor `linquant/reference.py` and `invariances.py` (MIT, attributed) into a test
  fixture; map our graph to `chunk_core(family=delta, rho=1, tau=1, eta=0, read_post=False, qz=none)`;
  random (non-one-hot) small inputs, no weights, no GPU. Complements the one-hot exact-F32 oracle,
  which proves row layout but not general numerics.
- **Done bar / correctness gate:** (1) ggml output vs fp64 reference within 1e-9 relative on the fp64
  path [tolerance from the handoff box; F32 ggml compare uses the existing NMSE gate]; (2) chunked graph
  equals the per-token path within the same tolerance; (3) the test runs in the normal CPU suite;
  (4) it FAILS on a seeded mutation of the WY prep and on a seeded mutation of the state pass.
- **Falsifier:** a seeded mutant that passes -> the oracle is not discriminating; not done.
- **Stop:** done bar met (answer); futility = fp64 reference cannot be mapped to our graph's
  parameterisation without changing semantics (record the mismatch, keep AKX-ALL-31.GDN only).
- **Expected effect:** none on speed; it is the correctness gate any later GDN kernel seed depends on.
- **Citation:** intake-2641#record, intake-2642#record.

## AK-QH-A -- audit output.weight / token_embd bytes per token on served DS41 / Q38FN GGUFs

- **Target:** DS41 and Q38FN served GGUFs; tensors `output.weight`, `token_embd.weight`; CPU lane.
- **change_class:** none (audit, gguf metadata read only).
- **Mechanism:** the output head is a dense `vocab x n_embd` GEMV read in full every decode token and
  never amortised; `token_embd` is a row gather (1 row/token), so it matters only when tied to the
  head or when its RAM footprint matters -- the audit must say which.
- **Expected effect derivation [inferred]:** head bytes/token = `n_vocab * n_embd * bpw/8`. Q6_K = 6.5625
  bpw, Q8_0 = 8.5, Q4_K = 4.5. Q6_K->Q4_K saves 1 - 4.5/6.5625 = 31.4% of head bytes; Q8_0->Q4_K saves
  47.1%. Decode is BW-bound on CPU, so expected tg gain ~= head_share * saving, where head_share =
  head bytes / total active bytes per token (to be audited, not assumed; for an MoE with small active
  set head_share is larger).
- **Output:** per model: current head/embd type, bytes/token, head_share, predicted tg gain.
- **Stop-on-futility:** predicted gain < 1% on a model (head already 4-bit, or head_share * saving <
  1%) -> that model's AK-QH-B arm is not run; record the null.
- **Citation:** intake-2646#record.

## AK-QH-B -- one Q4_K output-head conversion arm per surviving model

- **Prerequisite:** AK-QH-A predicts >= 1% for that model.
- **Target:** DS41 / Q38FN, CPU lane, C7r; `output.weight` re-quantized to Q4_K (all other tensors
  byte-identical); op path `ggml_mul_mat` GEMV on the head (iqk Q4_K path).
- **change_class:** `layout` (weight-format arm; no kernel source change -- if the loop requires a
  source manifest this arm runs as a model-variant A/B, not an authored candidate).
- **Expected effect:** AK-QH-A prediction [inferred]; bytes/token must fall by the audited head delta.
- **Correctness gate (quant change):** KLD vs the served GGUF and WikiText-2 PPL on the codified
  recipe, plus the six-task set at the existing floor; no task may regress beyond its recorded sigma.
- **Falsifier:** tg gain < half the predicted gain over valid interleaved pairs, or any quality gate
  fails.
- **Stop-on-answer:** gain within the prediction band with quality pass (KEEP candidate, still
  futility-only). **Stop-on-futility:** interim futility look shows the gain CI upper bound below 1%, or a
  quality gate fails -> record the null.
- **Citation:** intake-2646#record.

## AK-QR-1 -- offline activation rotation / permutation at a fixed per-32-block budget

- **Target:** one real Q4_K weight row (DS41 or Q38FN, a large-share GEMV) with captured activations;
  CPU, offline numpy-scale simulation; models the `q8_K` activation-quant window used by the CPU dot.
- **change_class:** none (offline probe; no kernel, no converter change).
- **Arms:** (0) no rotation (control); (1) 32-wide Hadamard; (2) magnitude-sorted channel permutation;
  (3) KLT composed with Hadamard.
- **Measures:** q8_K window error, block amax / clipping error, dKLD on the row output.
- **Pre-registered prediction (MambaQuant):** rotation has a positive sign and its effect shrinks as
  scale granularity narrows -- at 32-element blocks the effect may be near zero.
- **Expected effect [inferred]:** unknown magnitude; sign predicted positive, shrinking with granularity.
- **Correctness gate:** none on the model (offline); any later kernel/converter arm inherits KLD/PPL gates.
- **Stop-on-answer:** one arm beats control by more than the window-error noise (measured across
  repeated activation samples) -> it may seed a converter/kernel hypothesis (`arithmetic` or `layout`).
  **Stop-on-futility / falsifier:** no arm beats control beyond noise -> record the null.
- **Citation:** intake-2648#record, intake-2645#record.
