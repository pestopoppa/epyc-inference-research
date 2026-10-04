# coherence_gate: adoption and governance draft

Library: `scripts/lib/coherence_gate/` (schema `epyc.coherence_gate.v1`, `degeneracy.v2`,
`answer-graders.v1`). The API and rules are in the package docstrings; this page covers only what
adopting it requires. Origin: the operator approved it on 2026-10-04 to replace the copy-pasted INF-70
classifier. The evidence is in `/mnt/raid0/llm/tmp/q38t7-rescore/{AUDIT,RESCORE}.md`.

## 1. Governance: what needs ratification, and who ratifies

The library on its own is a tool. It needs no ratification to exist or to be used for exploratory
re-scoring.

Two kinds of adoption **do** change a measurement instrument. Both touch human-only trust-boundary paths,
so the **operator** ratifies them by running the ratify script themselves (agents prepare the package and
never apply it):

1. **An era row** in `epyc-orchestrator/orchestration/instrument_eras.yaml` (append-only). This is
   needed before any adopted verdict feeds a belief row, a promotion gate or a recipe gate. It covers
   INF-70 harness1, G2-CONC, PROD-1 and the Q38-T7 runner. The draft is in §1a.
2. **A protocol amendment** to Annex B (`measurement/protocols/bench-cpu.md`), with a one-line
   CHANGELOG entry in `MEASUREMENT.md`. This is needed **only** if the paired gate is made the mandatory
   meaning of "correctness/coherence" for promotion. The draft is in §1b.

   Adopting the library inside the AutoKernel evaluator bundle would be a third change, and it is
   **not recommended** (see §2, row 4). It would also need the kernel-research.md evaluator-bundle
   process, because denial 6 forbids changing the evaluator bundle outside it.

### 1a. Draft era row (not applied)

```yaml
  - id: OC1-coherence-gate-v1
    from: "<apply timestamp>"
    scope: output_coherence
    note: >
      Shared paired coherence gate (epyc-inference-research scripts/lib/coherence_gate @ <commit>;
      schema epyc.coherence_gate.v1, degeneracy.v2, answer-graders.v1) replaces the INF-70 classifier
      (inf70-degeneracy.v1 = tmp/inf70/agents/gdn-rowexact/classify.py == epyc-root
      scripts/inf70/harness1/classify.py, lib_gpublock.classify, ~15 client copies) in every gate
      that adopts it. One verdict per item (PASS | REGRESSION | BOTH_BAD | NEEDS_REVIEW) with the
      deciding tier (0 paired byte-identity, 1 ground truth + degeneracy.v2 paired, 2 injected judge);
      gate PASS | FAIL | INCOMPLETE, and INCOMPLETE is never PASS.
      PRE-BOUNDARY DEFECTS: INF-70 chat-path labels were computed on synthetic ids
      (tokens=list(range(n))), so uniq/top/run were vacuous and only words/ascii could fire.
      v1's `uniq < 0.35` is length-insensitive: every >= 1000-token reasoning trace is SALAD. And
      lib_gpublock read chat "stop" on 1-4-token answers as EARLY-EOS.
      RECONCILIATION: PAIRED byte-identity evidence retro-certifies, because tier 0 is unchanged.
      Absolute v1 labels (COHERENT/SALAD) are demote-to-prior as coherence evidence. They are
      re-scored by tail replay from banked full text (`python -m coherence_gate`, with the run's
      tokenizer or declared --text-only), never regenerated. Rows without stored full text
      (Q38-T7 20261004T025247Z phase B/C) cannot be replayed and stay priors.
      Post-boundary records are era-labelled by `schema` == epyc.coherence_gate.v1.
```

### 1b. Draft Annex B amendment (only if made mandatory for promotion; not applied)

> **B-COH-1 (amendment, <date>).** The *correctness/coherence* check named in this annex is a
> **paired** verdict. The candidate's greedy outputs are compared item by item with a named anchor's
> outputs on the same prompts by `coherence_gate` (schema `epyc.coherence_gate.v1`); inside AutoKernel
> it is the evaluator's own `compute_coherence`. A gate passes only on aggregate `PASS`. `FAIL` (any
> REGRESSION) fails the gate. `INCOMPLETE` (unjudged divergence, missing anchor output) blocks the gate
> and is never read as a pass. An absolute single-arm degeneracy label is not a coherence verdict.
> Degeneracy must be computed on real token ids, or on declared and recorded text-only surrogates;
> synthetic ids are refused. Every verdict record stores the full texts or their paths.

## 2. Migration list

| # | consumer | what it needs to adopt the library |
|---|---|---|
| 1 | **Q38-T7** `tmp/gpu-block-27b-20261003/q38_t7.py` + `lib_gpublock.py` (not in git) | Replace `import degeneracy as D`, `pair_a`, `grade` and `classify_text` with `coherence_gate.evaluate(nodraft_rows, dflash2_rows, truth, anchor={"arm": "speculative.n_max=0", "binary_version": "10303"})`. Rows already carry `text` + `token_ids` (from /tokenize) in the patched runner. Truth: question_pool rows (`grader_for` maps them), plus needle rows `{"grader": "needle", "expected": [code, name]}`. `verdict()` becomes the aggregate gate. Phase C streams have no no-draft pair: add one, or keep them on the documented single-arm `degeneracy.fails`. Retire `lib_gpublock.classify` (v1 lives on as `coherence_gate.classify_v1`). |
| 2 | **INF-70 harness1** `/workspace/scripts/inf70/harness1/{client.py:45, classify.py, analyze.py}` and the ~15 tmp client copies | Stop calling `classify(dict(tokens=list(range(npred)), ...))`, which the library refuses. Capture real ids: `/completion` returns `tokens`; on the chat path, `/tokenize` the content, or pass `--tokenizer`/`--tokenize-url`. Write rows with `id`, `text`, `token_ids`, `finish`, and a full `sha256` (the current `sha` is a 16-char prefix and is ignored). Replace per-arm labels with `python -m coherence_gate --base <anchor arm> --cand <arm> --truth question_pool.jsonl`. Mark `classify.py` frozen v1. The 403 banked `*.rows.jsonl` already match the row shape (a smoke run on speed-claim A1 vs B1 gave 12/24 identical, 0 regressions, gate INCOMPLETE on 12 unjudged MTP divergences). The tmp clients are historical; the era row covers them. |
| 3 | **G2-CONC** (`handoffs/active/cpu-decode-roofline-program.md` ~:4811, E-GATE Criterion 1 ~:4742, PROD-1 ~:652/:737; `autokernel-unified-surface-program.md:1255,2117`) | Not yet run, so this is a text change only, made by the owning session. Replace "classify by REASON COHERENT/SALAD/…" with "paired `coherence_gate` verdict vs the production anchor binary under the same `-np 4` admission and the ≥32-row probe; real ids from `/completion` `tokens`; gate PASS required, INCOMPLETE blocks". The v2 classes OK/DEGENERATE/SHORT/EARLY-EOS/EMPTY/HTTP-ERROR replace the v1 ones. Needs era row §1a. |
| 4 | **AutoKernel gates**: T0 `evaluator/correctness.py` `compute_coherence`, DS41 serving gate | **No swap.** `compute_coherence` is already paired, binds the anchor triple and the determinism class, and is stricter than tier 0. It sits in the immutable evaluator bundle (trust boundary, denial 6). Optional later step: report `coherence_gate.degeneracy` for divergent items, as an evaluator-bundle change ratified by the operator. |
| 5 | Cousins: DAR-LAT `critic_thread_gate.v2.py:148` (4-gram uniq ≥ 0.3, paired), batch-envelope `conc.py` (byte-identity), orchestrator `detect_repetition_loop`/`apply_garbage_gate` (runtime, not measurement) | No action is required. These are paired or runtime-only. Re-point them on their next version bump (`degeneracy.v2` loop trigger). |
| 6 | Docs that cite v1 as authoritative: `wiki/benchmark-methodology.md` ~:5240-5300 rule 5, `wiki/hardware-optimization.md` ~:5300 | Point them at this library on the next wiki compile. |
