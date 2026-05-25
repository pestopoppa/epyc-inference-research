# LongCat-Flash-Lite (n-gram Engram-family) on CPU — POC Negative Result

**Date**: 2026-05-25
**Tracks**: handoffs/active/engram-conditional-memory.md Track A Phases 3-6
**Verdict**: Family validated on CPU; specific checkpoint does not displace any current production role.

## What we tested

LongCat-Flash-Lite (Meituan, 68.5B total / 2.9-4.5B active, 31.4B in n-gram embedding tables, MIT-licensed). The only deployed Engram-family open-weight checkpoint at the time of this writeup. Production-quality GGUF available as `InquiringMinds-AI/LongCat-Flash-Lite-GGUF` (Q4_K_M = 37.4 GB). Served via a non-upstreamable Claude-Code-AI-generated fork at `https://github.com/InquiringMinds-AI/llama.cpp` branch `longcat-flash-ngram` head `56abe857d` (2026-04-27).

Architectural deviation from paper Engram (worth pinning here so the negative result on the checkpoint is not read as a negative on the paper architecture): LongCat injects at input embedding only (1-shot, no per-layer mid-stream injection), no scalar sigmoid gate, no depthwise causal conv. Just `embed = base / 13 + Σ post_proj(table[n,k][hash(...)])`. Polynomial rolling hash (not multiplicative-XOR), 4 hash heads per order (not 8), n ∈ {2,3,4} (paper uses {2,3}), custom 131k tokenizer with no canonicalization map. So Track A validates the *family*, not the *paper-faithful architecture*.

## Hardware + protocol

- EPYC 9655 single socket, 1.1 TB DDR5 NPS-default (~460 GB/s aggregate, validated)
- Uptime 6 days, performance governor, `drop_caches` issued before bench, host idle (no production stack contention)
- Bench env: `OMP_PROC_BIND=spread OMP_PLACES=cores OMP_WAIT_POLICY=active KMP_BLOCKTIME=10 numactl --interleave=all` (per `feedback_omp_env_stack_required.md`)
- `llama-bench -t 96 -fa 1 -r 1 -p 512,4096 -n 128` (single repetition; rationale: we're at gate-pass-or-fail granularity, not optimizing)
- Quality eval: `sentinel_questions.yaml` (39 questions, 11 suites) via /v1/chat/completions, `temperature=0 max_tokens=2048 timeout=480s`
- LongCat served via InquiringMinds fork at `/mnt/raid0/llm/llama.cpp-longcat-probe` (worktree, branch `probe/longcat-build`)
- gemma4-26B-A4B Q4_K_M MTP served via `ik_llama.cpp` at production parameters (MTP drafter, `-ctk q8_0 -ctv q8_0 -ub 512 --no-mmap`)

## Speed (Phase 4)

| Test | LongCat-Lite Q4_K_M (InquiringMinds fork) | gemma4-26B-A4B Q4_K_M no-MTP (ik_llama) | gemma4-26B-A4B Q4_K_M MTP (production) | Qwen3.6-35B-A3B Q8_0 (production llama.cpp) |
|------|------:|------:|------:|------:|
| pp512  | 322.65 | 957.82 | (not benched isolated) | 439.31 |
| pp4096 | 258.33 | 891.39 | (not benched isolated) | 435.51 |
| tg128  | **37.08** | 47.71 | **~76.5** (memory ref) | 25.17 |

LongCat decode is **−22% vs gemma4 no-MTP**, **−51% vs gemma4-MTP production**, **+47% vs Qwen3.6 frontdoor**. Above the speed-gate abandon threshold (15 t/s) and above the speed-gate proceed threshold (35 t/s) — passes the speed limb of Gate A in isolation. But the apples-to-apples worker comparator is gemma4-MTP at 76 t/s, which LongCat halves.

## Quality (Phase 5)

39-question sentinel set, identical scoring per `scoring_method` field on each question.

| Suite (count) | LongCat-Lite | gemma4-MTP | Δ |
|---|---:|---:|---:|
| agentic (3) | 1/3 = 33.3% | 1/3 = 33.3% | tie (both weak) |
| coder (3) | 3/3 = 100% | 3/3 = 100% | tie |
| general (4) | 4/4 = 100% | 4/4 = 100% | tie |
| gpqa (4) | 1/4 = 25% | 2/4 = 50% | +1 gemma4 |
| hotpotqa (4) | 3/4 = 75% | 1/4 = 25% | **+2 LongCat** |
| instruction_precision (4) | 3/4 = 75% | 4/4 = 100% | +1 gemma4 |
| long_context (1) | 1/1 | 1/1 | tie |
| math (6) | **0/6 = 0%** | **4/6 = 67%** | **+4 gemma4** |
| mode_advantage_hard (3) | 2/3 = 67% | 2/3 = 67% | tie |
| simpleqa (3) | 1/3 = 33% | 1/3 = 33% | tie (different questions) |
| thinking (4) | 2/4 = 50% | 3/4 = 75% | +1 gemma4 |
| **TOTAL (39)** | **21/39 = 53.8%** | **26/39 = 66.7%** | **+12.9pp gemma4** |

Key observations:

- **Math 0/6 was confirmed structural** (not max_tokens=512 truncation). First run at max_tokens=512 = 0/6; rerun at max_tokens=2048 = still 0/6, with multiple questions running the full 2048-token budget without converging. Sample failure mode: `sentinel_math_01` — LongCat correctly identified `3^4 = 81` but then wrote `4^3 = 4×4×4 = 16` (genuine arithmetic error), then `8 + 16 = 18` (compounded). This is not a truncation artifact — it's the model going off the rails inside its own reasoning chain.
- **LongCat's hotpotqa edge (3/4 vs gemma4's 1/4)** is the one real win. The n-gram-augmented embedding may help with literal-string multi-hop retrieval where the answer-bearing entity name reappears across the prompt. Single suite, n=4, anecdotal — not enough to overturn the math + instruction + gpqa losses.
- **Both models tie at 33% on agentic** — the deployed worker is genuinely weak here too, which makes this a non-differentiator. (It also means our agentic-routing improvements are NOT capped by worker quality on these sentinel tasks specifically.)
- **Discrepancy with Meituan's published numbers**: Meituan reports MATH500 = 96.80%, AIME24 = 72.19%. Our sentinel math is 0/6. Three plausible explanations: (a) sentinel math questions are framing-different from MATH500 / exact-match scoring is unforgiving of LongCat's verbose answer format; (b) `temperature=0` produces degenerate outputs on this model where higher temperature would not; (c) the chat-template warning at server startup ("Neither string content nor typed content is supported by the template") indicates a tokenization mismatch that subtly degrades reasoning. Not investigated further — even fixing it would still leave LongCat behind on the other suites.

## Decision (Phase 6)

**Track A — CLOSED, NEGATIVE.**

LongCat-Flash-Lite Q4_K_M does not pass Gate A as a candidate for any current production role:

| Role | LongCat decode vs deployed | LongCat quality vs deployed | Verdict |
|---|---|---|---|
| worker_general (gemma4-MTP) | −51% (37 vs 76) | −12.9pp on sentinel | dominated on both axes |
| frontdoor (Qwen3.6-35B-A3B Q8) | +47% (37 vs 25) | sentinel-eval skipped but agentic 1/3 = 33% is a known weakness; frontdoor requires strong agentic | quality regression unacceptable for this role |
| ingest_long_context, architect_general | mismatched param scale / quant — not benchmarked | n/a | not a candidate |

Decision flow followed the handoff's Sequencing Decision diagram: Gate A FAIL → archive Track A. Do not proceed to use the InquiringMinds fork for any production deployment.

**What we DO take away from Track A** — three positive findings worth preserving:

1. **N-gram-augmented MoE inference works on CPU at production-relevant rates.** 37 t/s decode on 68B/4.5B-active Q4_K_M with a 31.4B-param n-gram embedding table fully resident in DDR5 is a real number. This validates the *family*'s CPU feasibility (the bandwidth math from the CXL follow-up paper said ~10 KB/token at FP8 → <0.2% of our DDR5 aggregate — measured speed reflects model FFN + MoE pressure, not lookup pressure).
2. **The InquiringMinds llama.cpp fork works.** A non-upstreamable AI-generated patch is enough to run a novel architecture end-to-end on our stack. Useful precedent if a paper-faithful Engram checkpoint ever lands.
3. **The "math 0/6 from truncation" hypothesis was wrong.** We re-ran at 4× the token budget and the score was identical. Whenever a model shows a sharp categorical failure on one suite of a heterogeneous eval, look for structural reasons before assuming budget — and confirm by varying the budget. This is a small reusable lesson for future Track-A-style evals.

**Track B is unaffected** by this result. Track B targets paper-faithful Engram on a backbone we train ourselves (Qwen3.6 / gemma4 / SmolLM proxy); LongCat is a different architecture (input-only n-gram add, no gating, no conv). The Phase 0a engram-spike package (identity-at-step-zero invariant verified) and the Phase 0b GPU-proxy plan stay in force.

## Artifacts

- Speed bench logs: `/tmp/longcat-bench.log`, `/tmp/qwen36-bench.log`, `/tmp/gemma4-bench.log`
- Eval results JSON: `/mnt/raid0/llm/epyc-inference-research/research/engram-spike/eval/longcat-results-2048.json` (LongCat), `gemma4-results.json` (gemma4)
- Eval driver: `/mnt/raid0/llm/epyc-inference-research/research/engram-spike/eval/run_eval.py`
- LongCat GGUF: `/mnt/raid0/llm/models/longcat-flash-lite-q4km/LongCat-Flash-Lite-Q4_K_M.gguf` (kept on disk in case we ever want to re-evaluate under different conditions; 37.46 GiB)
- llama.cpp fork: `/mnt/raid0/llm/llama.cpp-longcat-probe` (git worktree on `probe/longcat-build` tracking `inq/longcat-flash-ngram`)

## What we did NOT do

- Did not bench LongCat on the 256K-context YaRN setting (handoff Phase 4 1K/4K/16K only). The Engram bandwidth story should scale gracefully but this is unmeasured on our setup.
- Did not run the FP8 / FP16 ablation on the n-gram tables; only Q4_K_M tested. The InquiringMinds GGUF publisher specifically warns that quality degrades sharply ≤Q3, suggesting embedding-table quant sensitivity that a higher-precision run might mitigate. Out of scope for the gate decision since even at Q4_K_M LongCat is dominated by gemma4 — improving its quality would have to overcome a 12.9pp deficit AND a 2× speed deficit to be worth deploying.
- Did not test alternative chat templates / system prompts / temperatures that might fix the math behavior. The Meituan-vs-our gap there is interesting and worth chasing if anyone needs LongCat for a math-routed role specifically — but for the umbrella worker_general comparison, the verdict stands.
- Did not test LongCat's draft model for spec-dec compatibility with gemma4 or Qwen3.6 backbones — different tokenizers make this a non-starter without retraining.
