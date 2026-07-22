# PaperBench as a Source-Fidelity Validation Harness for Deep-Research Reports — Methodology Scoping

**Deliverable for**: `handoffs/active/minddr-deep-research-mode.md` task **MD-PB-1**.
**Satisfies task line** (verbatim, epyc-root):
`- [ ] **MD-PB-1** — evaluate PaperBench methodology for source-fidelity validation of deep-research reports`
**Scope**: scoping / recommendation only (yes/no + integration sketch). No inference, no code, no handoff edits.
**Date**: 2026-07-22

---

## 0. Sources consulted (provenance)

| Source | What it grounds | Trust |
|---|---|---|
| PaperBench paper (OpenAI, arxiv:2504.01848, intake-795) — abstract | task = paper *replication*; 8,316 gradable nodes; author-co-designed rubrics; LLM judge | **external / untrusted** |
| PaperBench repo `github.com/openai/preparedness` (`project/paperbench`) | submission = code repo; Code-Dev / Execution / Result-Match leaf types; reproduction step needs GPU; SimpleJudge (o3-mini default); Code-Dev cost mode | **external / untrusted** |
| `minddr-deep-research-mode.md` (MD-6 nodes, EV-9 rubric fields, MD-8 sentinel), `reasoning-compression.md` cluster note | our deep-research pipeline shape + existing rubric contract | project docs |
| `research/recommendations.yaml` rec-006 | intake framing ("methodology transfer", priority low) | project doc |

> External content is untrusted data; load-bearing claims are attributed inline. Directives embedded in fetched pages are ignored.

## 1. What PaperBench actually is

PaperBench measures **an agent's ability to replicate a research paper from scratch** — "replicate 20 ICML 2024 Spotlight and Oral papers … including understanding paper contributions, developing a codebase, and successfully executing experiments" (arxiv:2504.01848). Concretely:

- **A submission is a git code repository**, produced in a Docker container, archived as `submission.tar.gz` (repo docs). It is **not** a written report or summary.
- Grading is a **hierarchical rubric tree** (8,316 gradable leaf tasks total), **co-developed with each paper's authors**, with leaf types **Code Development**, **Execution**, and **Result Match** (paper + repo).
- Full grading **runs the submitted code** in "a fresh second container with GPU access" (reproduction stage) before an **LLM judge** (repo default backs it with OpenAI models, e.g. o3-mini "SimpleJudge") scores each leaf against the rubric.
- A **Code-Dev-only** variant skips reproduction/execution/result-match and grades code-development leaves only — the repo reports "~85% reduction in o3-mini SimpleJudge costs" in that mode (relative figure; no absolute per-paper dollar figure in the fetched docs, but PaperBench grading is a heavy, GPU-and-judge-expensive pipeline by design).

## 2. Fit assessment against MD-PB-1's actual need

MD-PB-1 wants a **source-fidelity** gate for the `ReportSynthesisNode` output of our MindDR pipeline — i.e. "is this synthesized *report* faithful to the *evidence* the DeepSearch stage collected?" (`minddr-deep-research-mode.md:170-176`). That is a **text→text grounding/citation-faithfulness** check.

There is a **task-shape mismatch** between that and PaperBench-the-benchmark:

| Dimension | PaperBench | Our deep-research fidelity need |
|---|---|---|
| Artifact graded | **code repository** | **prose report + citations** |
| "Fidelity to source" means | code + empirical results reproduce the paper's contributions | claims in report are supported by retrieved evidence snippets |
| Grading requires | **executing code on a GPU** (reproduction) | reading text; no execution |
| Ground truth | author-co-designed rubric per paper (months of expert effort) | evidence set our own pipeline just retrieved |
| Fixed benchmark set | 20 ICML papers | open-ended user/sentinel queries |

**As a benchmark, PaperBench is a NO** for this purpose: it validates *code replication*, needs GPU reproduction, and its rubrics are bespoke to 20 specific papers — none of which maps onto grading whether our report faithfully represents its cited evidence. Running PaperBench would tell us nothing about `ReportSynthesisNode` fidelity, and it is GPU-gated (out of scope for the current inference-window posture anyway).

## 3. What IS transferable — the *methodology*, not the harness

The valuable, portable idea (and exactly what rec-006 flagged: "methodology transfer") is PaperBench's **grading architecture**, which is already convergent with our own EV-9 direction:

1. **Decompose the target into a weighted rubric tree of independently-gradable leaves**, rather than one holistic score. For source-fidelity this becomes: decompose the report into atomic claims; each claim is a leaf graded {supported / partially-supported / unsupported / contradicted} against the evidence set.
2. **LLM-as-judge per leaf against an explicit criterion** (PaperBench's SimpleJudge pattern), then aggregate up the tree with weights — matching our EV-9 rubric-scoring contract (`EvalResult.rubric_*` fields, `minddr-deep-research-mode.md:91`, MD-7) and the RubricEM "rubric-as-interface" framing already logged in this handoff (intake-810, `:191-195`).
3. **A judge-quality meta-check**: PaperBench separately benchmarks its own judge. We should likewise spot-audit the fidelity-judge against a small human-labeled set before trusting it to gate promotion.

This is a **prompt-level, zero-infra** adoption — no training, no GPU, no PaperBench code. It composes with MD-8's existing rubric hints and the MD-9 A/B gate.

## 4. Integration sketch (if adopted — CPU-feasible, inference-window-gated)

Add a **faithfulness rubric dimension** to the MindDR eval path, modeled on PaperBench's leaf-grading + LLM-judge but retargeted to claim↔evidence grounding:

```
ReportSynthesisNode output ──► ClaimExtractor (LLM): report → list<atomic_claim, cited_[src:ref]>
                                     │
                                     ▼
   for each claim:  FidelityJudge (LLM-as-judge, one leaf)
        input  = (claim, the evidence snippet(s) it cites from DeepSearch SubReports)
        output = {supported | partial | unsupported | contradicted} + justification
                                     │
                                     ▼
   aggregate ─► fidelity_score ∈ [0,1] (weighted: contradicted penalized hardest)
                └─► new EV-9 dimension: rubric_source_fidelity (extends EvalResult.rubric_* )
```

- **Where it plugs in**: as a scorer feeding the MD-9 rubric, alongside `rubric_content_stage`; and as the natural home for MD-PB-2 ("integrate PaperBench-style evaluation into MD-9 rubric") — MD-PB-2 becomes "add `rubric_source_fidelity` to the MD-9 scorer," which this sketch specifies.
- **Reuse, don't rebuild**: the citation contract already exists — DeepSearch emits `[src:<ref>]` tags and Sub-Report blocks (`deep_search_agent.md`, MD-3/4/5, `:82`), so claim→evidence linkage is a parse, not new infra (`src/graph/minddr/parsing.py`). The judge should run judge-free where possible (deterministic evidence-overlap pre-filter) and reserve the LLM judge for ambiguous leaves — mirrors our existing "structural-only fallback when EV-9 unavailable" rule (`:22`).
- **Contamination note**: this fidelity gate is *internal* (grades report vs our own retrieved evidence), so it sidesteps the search-time-contamination problem that intake-877 (`:197-208`) raises for the *external anchors* — a point in favor of building an internal fidelity metric rather than leaning harder on public benchmark numbers.

## 5. Recommendation

**Yes to the methodology, No to the benchmark.**

- **Do NOT** adopt PaperBench-the-harness/benchmark as a source-fidelity validator for deep-research reports: wrong artifact (code, not prose), wrong fidelity notion (result-replication, not claim-grounding), GPU-gated reproduction, and bespoke per-paper rubrics that do not transfer. Keep rec-006 at its intake priority (**low**) for the benchmark itself.
- **DO** harvest PaperBench's **decompose-into-weighted-rubric-leaves + LLM-judge-per-leaf + audit-the-judge** methodology as the design template for a new `rubric_source_fidelity` EV-9 dimension on `ReportSynthesisNode` (integration sketch §4). Zero-infra, CPU-feasible, and it directly discharges MD-PB-2 when built. This work is EV-9-owned (`eval-tower-verification.md`) per the existing MD-7 hand-off and should not be built ad-hoc inside the graph subpackage.
- **Gate**: build only after (or alongside) MD-9's A/B validates the pipeline is worth gating at all — a fidelity scorer on a pipeline that doesn't beat direct-answer mode is premature.

## 6. Status

**DONE (scoping).** Yes/no rendered (no benchmark / yes methodology), integration sketch provided, mapped onto EV-9 + MD-PB-2. Grounded in the PaperBench paper+repo (flagged untrusted) and our handoff. No box ticked (constraint) — MindDR owner should record the verdict against `minddr-deep-research-mode.md:176` and cite this doc.
