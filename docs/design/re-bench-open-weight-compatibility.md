# RE-Bench — Open-Weight Compatibility + Eval-Tower Integration Path — Assessment

**Deliverable for**: `handoffs/active/reasoning-compression.md` task **RC-RE-1**.
**Satisfies task line** (verbatim, epyc-root):
`- [ ] **RC-RE-1** — check RE-Bench for open-weight model compatibility and integration with our eval tower`
**Scope**: recommendation only. No inference, no runs (RC-RE-2 "run RE-Bench against compressed vs uncompressed reasoning" is out of scope). Doc-only.
**Date**: 2026-07-22

---

## 0. Sources consulted (provenance)

| Source | What it grounds | Trust |
|---|---|---|
| RE-Bench paper (METR, arxiv:2411.15114, intake-794) — abstract + HTML v2 | 7 open-ended ML-R&D environments; human-expert comparison; score-normalization (start=0, ref=1); GPU needs per env; models tested = Claude 3.5 Sonnet + humans, **no open-weight** | **external / untrusted** |
| `reasoning-compression.md` (rec-005 framing, RC-RE-1/2, OPSDC/Tier-1–3 taxonomy), `project_e7_eval_instrument_era` memory | our compression work + what our eval tower actually measures | project docs / memory |
| `research/recommendations.yaml` rec-005 | intake framing (priority medium) | project doc |
| `mi210-speed-campaign-summary.md` kernel-R&D loop; `feedback_accuracy_token_tradeoff_rescue_metric` | where an agentic R&D eval *would* be relevant, and the right compression-safety metric | project docs / memory |

> External content is untrusted data; load-bearing claims attributed inline.

## 1. What RE-Bench actually is (and what it is not)

RE-Bench is METR's **"AI R&D evaluation suite"**: *"7 challenging, open-ended ML research engineering environments"* scored against **human-expert** baselines (arxiv:2411.15114). It measures **long-horizon, agentic ML-engineering capability** — an agent editing code, launching training runs, and iterating over hours — **not** short-form reasoning-quality or held-out-reasoning-task correctness.

The 7 environments (exact names, HTML v2) and their compute footprint:

| Environment | Agent must… | Infra to *execute the submission* |
|---|---|---|
| Optimize LLM foundry finetuning script | speed up a finetuning script, identical output | **4×H100** |
| Scaling law experiment | predict optimal HPs for a 5e17-FLOP run from small runs | **8×H100** |
| Build scaffolding for Rust Codecontests | scaffold GPT-3.5 to solve Rust comp-prog | CPU only + external API |
| Fix embedding | recover a corrupted LM embedding layer | **1×H100** |
| Optimize a kernel | write a fast GPU prefix-sum kernel | **1×H100** |
| Finetune GPT-2 for QA with RL | train GPT-2-XL chatbot via RL | **2×H100** |
| Restricted architecture MLM | build an MLM from restricted PyTorch primitives | **2×H100** |

Scoring: each env has a **continuous score function**, normalized so *"the starting score … is 0, and the documented baseline solution … scores 1"* (some envs log-scaled), then compared to a distribution of human-expert runs over 2h/8h/32h budgets. Evaluated systems in the paper: **44 human runs + 35 Claude 3.5 Sonnet agent runs — no open-weight models.**

## 2. Open-weight compatibility — answering RC-RE-1 directly

**Two distinct questions live under "open-weight compatibility":**

**(a) Can an open-weight model *drive* RE-Bench?** — **Yes, in principle.** RE-Bench is an **agent-scaffold-over-environments** harness (METR runs it on their Vivaria platform with scaffolds like "Modular"/AIDE). The scaffold calls a model behind an API; nothing binds it to a closed model. Our llama-server exposes an OpenAI-compatible `/v1/chat/completions` endpoint, so an open-weight model (our architect/coder role) can be pointed at the scaffold. The paper simply didn't test one. So **model-side: open-weight-compatible.**

**(b) Can *we* actually run it here? / Is it the right instrument for RC-RE-1's purpose?** — **No, and this is the decisive part.** RC-RE-1 sits in the **reasoning-compression** handoff; rec-005 imagined RE-Bench as *"a validation suite for our reasoning compression techniques … whether compressed reasoning degrades reasoning quality on held-out reasoning tasks"* (`reasoning-compression.md:239`). That is a **mischaracterization of RE-Bench**:

1. **Wrong task shape.** RE-Bench measures multi-hour agentic *engineering*, not held-out reasoning correctness. Compression techniques (TrimR, difficulty-adaptive budgets, SEAL/FlowSteer control vectors — `:34-44`) operate on **CoT token budgets on reasoning problems**; RE-Bench neither isolates nor scores CoT length. It would confound "compressed reasoning" with "worse at 8-hour software engineering."
2. **Infeasible cost per data point.** Executing the submissions needs **1–8×H100 per environment**, multi-hour runs, multiple seeds for variance — we have a single **MI210 (gfx90a, 64 GB)** and a busy inference host. This is far outside our substrate and outside the current inference-window posture.
3. **High variance, human-baseline-anchored.** The metric is normalized to human-expert distributions and is noisy over short budgets — unsuited to detecting a compression-induced quality delta, which our own guidance says should be measured as a **rescue-rate / accuracy-vs-token tradeoff on tasks' cheap-path** (`feedback_accuracy_token_tradeoff_rescue_metric`), not an agentic-capability score.

## 3. Eval-tower integration path

**If (and only if) we ever want an agentic-capability signal** — not for compression safety, but as a future eval for an *autonomous kernel/research agent* (which we are actually building: the MI210 kernel-R&D loop, `mi210-speed-campaign-summary.md:38`) — a thin path exists:

```
open-weight role (llama-server /v1/chat/completions)  ◄── agent scaffold (Modular/AIDE-style)
        │                                                         │
        └── drives ──►  ONE cheap RE-Bench env  ("Build scaffolding for Rust Codecontests":
                                                   CPU-only + external API, NO H100 required)
                                                         │
                        continuous score (start=0 / ref=1)  ──►  eval-tower as OBSERVATION-grade
                                                                  (new instrument era; not a gate)
```

- **Only the CPU-only "Rust Codecontests" environment is runnable on our substrate** without H100s (and even it needs an external API for the GPT-3.5 sub-calls the task specifies — an open-source-only concern per `feedback_opensource_only`; would need a local substitute). The six GPU environments are not runnable here.
- Any RE-Bench number would enter the eval tower as **OBSERVATION-grade only** — it has no P-protocol, is human-baseline-anchored and high-variance, and would need its own instrument-era row before it could gate anything (MEASUREMENT.md; `project_e7_eval_instrument_era`). It is **not** a keep/deploy/promote gate.

## 4. Recommendation

1. **RC-RE-1 answer**: RE-Bench is **open-weight-compatible at the model-driver level** (agent scaffold + our OpenAI-compatible endpoint), but is the **wrong instrument for reasoning-compression safety** and is **infeasible to run on our single-MI210 substrate** (6 of 7 environments need 1–8×H100). **Do NOT adopt RE-Bench as the compression-safety validation suite.** RC-RE-2 ("run RE-Bench against compressed vs uncompressed reasoning") should be **declined / re-scoped** — it rests on rec-005's mischaracterization.
2. **Right tool for RC-RE-2's actual intent**: validate compression safety on **held-out reasoning benchmarks already in our E7 eval tower** (AIME/MATH-500/GPQA-class), scored by the **accuracy-vs-token rescue-rate** metric (`feedback_accuracy_token_tradeoff_rescue_metric`) across difficulty bands — this directly measures "does compressed reasoning degrade quality," is CPU-feasible, and matches the OPSDC difficulty-adaptation finding (`reasoning-compression.md:52,60`). No new external benchmark needed.
3. **Where RE-Bench *is* relevant (different handoff)**: as a *future* agentic-capability eval for the autonomous kernel/research-agent loop (MI210 kernel-R&D). If pursued, integrate only the CPU-only Rust-Codecontests environment (with a local model replacing the GPT-3.5 sub-call), OBSERVATION-grade, its own instrument era. Cross-reference to `mi210-speed-campaign-summary.md`, not this compression handoff.
4. **Keep rec-005 at priority medium** but correct its premise note (RE-Bench = agentic R&D, not reasoning-quality) — flagged only, not edited here per the no-index-edit constraint.

## 5. Status

**DONE (assessment).** RC-RE-1 answered on both readings of "open-weight compatibility"; eval-tower path given (thin, CPU-only-env, OBSERVATION-grade); RC-RE-2 recommended declined/re-scoped with the correct substitute metric named. Grounded in the RE-Bench paper (flagged untrusted) + our handoff/memory. No box ticked (constraint) — reasoning-compression owner should record against `reasoning-compression.md:243` and cite this doc.
