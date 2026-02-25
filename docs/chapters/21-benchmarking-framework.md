# Chapter 21: Benchmarking Framework

## Introduction

We developed an 8-suite benchmarking framework to evaluate models for specific roles in our orchestration system. Unlike generic benchmarks (MMLU, etc.), our suites test task-specific capabilities: can a model follow precise formatting? Can it chain multi-step reasoning? Can it generate valid tool calls?

**Key Achievement**: 61 baseline models evaluated, with 381 total configurations (including MoE/speculative variants).

## The 8 Benchmark Suites

Every model entering the orchestration system is measured against eight purpose-built suites, each targeting a specific capability that maps directly to an agent role. This is not about leaderboard scores -- it is about answering "can this model do the job we need it to do?"

<details>
<summary>Suite definitions and role mappings</summary>

| Suite | Purpose | Key Test | Role Placement |
|-------|---------|----------|----------------|
| **Thinking** | Chain-of-thought reasoning | Multi-step logical deduction | oracle_reasoning, architect |
| **Coder** | Code generation & debugging | Working code with edge cases | coder_escalation |
| **Math** | Mathematical reasoning | Step-by-step proofs | Qwen2.5-Math for invariants |
| **General** | Instruction following | Summarization, reformatting | worker_general |
| **Agentic** | Tool calling | Valid JSON function calls | frontdoor, orchestrator |
| **VL** | Vision-language | OCR, image understanding | worker_vision |
| **Long Context** | Information retrieval | Needle-in-haystack (4K-50K tokens) | ingest_long_context |
| **Instruction Precision** | Format compliance | Exact output structure | **Critical for orchestration** |

</details>

## Claude-as-Judge Scoring

We use Claude as an independent judge rather than algorithmic rubrics. Early experiments showed algorithmic scoring severely underscored models (38% vs 89% for the same output) because pattern matching breaks on unexpected but correct formats. Claude understands semantics, awards partial credit, and stays consistent across hundreds of evaluations.

<details>
<summary>Scoring rubric and rationale</summary>

### Scoring Rubric

| Score | Meaning |
|-------|---------|
| 3 | Correct answer with good reasoning |
| 2 | Partially correct or truncated |
| 1 | Wrong but reasonable attempt |
| 0 | Completely wrong, empty, or garbage |

### Why Claude-as-Judge?

- **Semantic understanding**: Recognizes correct answers in unexpected formats
- **Partial credit**: Awards 2 for "right approach, minor error"
- **Consistency**: Same model judges all, eliminating evaluator variance
- **Scalability**: Can score hundreds of responses efficiently

</details>

## Benchmark Hardening (December 2025)

Initial benchmarks had ceiling effects -- top models scored 89-93%, making it impossible to differentiate them. We hardened all suites by bumping every tier up one difficulty level and introducing post-doctoral-level T3 questions. After hardening, no model hits 90%+ and the score distribution spreads meaningfully across model classes.

<details>
<summary>Hardening details and expected score distributions</summary>

| Change | Before | After |
|--------|--------|-------|
| T1 questions | Easy | Medium (relabeled from T2) |
| T2 questions | Medium | Hard (relabeled from T3) |
| T3 questions | Hard | Post-doctoral level |

**New T3 Examples**:
- Thinking: Causal inference DAGs (collider bias)
- Math: Prove E[N] = e where S_n > 1 for uniform sum
- Coder: Lock-free stack ABA problem
- Agentic: Multi-agent coordination under time budget

### Expected Score Distribution (Post-Hardening)

| Model Class | Expected Score |
|-------------|----------------|
| 0.5B-1.5B draft models | 30-50% |
| 4B-8B general models | 50-70% |
| 8B+ specialized thinking | 60-80% |
| 14B+ large models | 70-85% |

Top models no longer hit 90%+ ceiling.

</details>

## Running Benchmarks

You can run the full benchmark pipeline or target individual suites. Results land in a structured directory that separates raw outputs from Claude-as-Judge review scores, making it easy to compare across models and configurations.

<details>
<summary>Commands and results layout</summary>

### Full Suite

```bash
./scripts/benchmark/run_overnight_benchmark_suite.sh --suite all
```

### Specific Suite

```bash
./scripts/benchmark/run_overnight_benchmark_suite.sh --suite thinking
./scripts/benchmark/run_overnight_benchmark_suite.sh --suite coder
./scripts/benchmark/run_overnight_benchmark_suite.sh --suite instruction_precision
```

### Results Location

```
benchmarks/
├── prompts/
│   ├── v1/              # Rubric-scored (Claude-as-Judge)
│   └── debug/           # Deterministic scoring (see Ch24)
├── results/
│   ├── runs/            # Raw benchmark outputs
│   ├── reviews/         # Claude-as-Judge scores
│   │   ├── {model}_baseline.csv
│   │   └── summary.csv
│   └── index.jsonl      # Benchmark index for comparison
```

**Note:** Benchmark prompts are gitignored — they are reconstructible from public sources. See [Chapter 24: Benchmark Suite Construction](24-benchmark-suite-construction.md) for the construction methodology and reconstruction instructions.

</details>

## Instruction Precision Suite

Models that fail instruction precision break TaskIR parsing, which means the entire orchestration pipeline falls apart. This suite is the gate that decides whether a model can even be considered for orchestration roles like frontdoor or dispatcher -- anything below 70% is disqualified.

<details>
<summary>Test matrix and role gate</summary>

| Test | What It Checks |
|------|----------------|
| Exact JSON structure | Can emit valid JSON with required fields |
| Format preservation | Respects specified output format |
| Constraint compliance | Follows "do not" instructions |
| Self-referential accuracy | Can accurately describe own output |

**Role Gate**: Models scoring <70% on instruction precision are not considered for orchestration roles (frontdoor, dispatcher).

</details>

## Quality vs Speed Trade-offs

Benchmarks capture both quality scores and speed per question, which lets you see exactly what you give up (or do not) when adding acceleration. The key finding: speculative decoding gives you a 10x speed boost with zero quality loss because it uses the same model, while MoE reduction trades quality for speed in a predictable curve.

<details>
<summary>Configuration comparison data</summary>

| Configuration | Quality | Speed | Use Case |
|---------------|---------|-------|----------|
| Qwen2.5-Coder-32B baseline | 89% | 2.89 t/s | Quality critical |
| Qwen2.5-Coder-32B + spec decode | 89% | 28.79 t/s | **Best balance** |
| Qwen3-Coder-30B + MoE4 | 85% | 33.6 t/s | Speed critical |
| Qwen3-Coder-30B + MoE3 | 78% | 37.7 t/s | Speed extreme |

**Key Insight**: Speculative decoding preserves quality (same model). MoE reduction trades quality for speed.

</details>

## Orchestrator Benchmarks

The orchestrator benchmark pipeline compares orchestrated responses against direct large-model baselines, measuring whether the multi-agent routing system retains quality while delivering speed gains. It runs in four phases -- smoke, compare, optimize, verify -- and now sources most of its questions live from HuggingFace datasets rather than static YAML files.

<details>
<summary>Scripts, datasets, and CLI options</summary>

### Scripts

| Script | Purpose |
|--------|---------|
| `scripts/benchmark/run_orchestrator_benchmark.py` | Full 4-phase benchmark runner (smoke, compare, optimize, verify) |
| `scripts/benchmark/compare_orchestrator_direct.py` | Per-suite orchestrator vs baseline comparison |

### On-the-Fly Dataset Sampling (January--February 2026)

Nine suites now sample fresh questions from real HuggingFace datasets on each run, totaling 35,560+ questions:

| Suite | Dataset(s) | Pool Size |
|-------|-----------|-----------|
| general | MMLU (cais/mmlu) | 14,042 |
| math | GSM8K + MATH-500 | 1,819 |
| coder | HumanEval + MBPP | 664 |
| thinking | ARC-Challenge + HellaSwag | 11,214 |
| instruction_precision | IFEval (google/IFEval) | 541 |
| vl | OCRBench + ChartQA | 3,500 |
| gaia | GAIA (gaia-benchmark/GAIA) | 165 |
| cruxeval | CRUXEval (cruxeval-org/cruxeval) | 1,600 |
| bigcodebench | BigCodeBench (bigcode/bigcodebench) | 1,140 |

Adapters in `scripts/benchmark/dataset_adapters.py`. Falls back to static YAML for `agentic`, `long_context`, and `mode_advantage`.

### Mode-Advantage Suite (February 2026)

A dedicated `mode_advantage` suite (90 YAML questions) specifically designed to produce strong routing signal for MemRL. Unlike existing suites where all tasks are solvable by direct inference, mode-advantage tasks structurally require specific execution modes:

| Category | Count | Mode Signal | Example |
|----------|-------|-------------|---------|
| Computation-gated | 15 | react >> direct | `7^13 mod 97` — model hallucinates, Python gets 38 |
| Iterative-fix | 15 | repl >> direct | Bug fix with test suite — REPL runs tests iteratively |
| Multi-step composition | 15 | delegation >> direct | Chained calculations requiring 3+ sequential steps |
| Escalation-gated | 15 | specialist >> frontdoor | A*, trie, Union-Find — 30B can't decompose |
| Mini-SWE | 30 | repl+delegation >> direct | Broken code + failing tests + known fix |

This shifts MemRL reward distribution from ~5% specialist-wins (+1.0) to ~25-35%, enabling the router to learn *when* to route, not just *that* routing has a cost.

**Stratified sampling**: `--stratify-tiers` draws equal questions per difficulty tier for suites with real tier metadata (MMLU, Math, IFEval). Other suites silently fall through to uniform random.

<details>
<summary>Code: CLI options</summary>

```bash
# Run Phase 2 (comparison) with API restart
./run_orchestrator_benchmark.py --phase 2 --restart-api

# Compare single suite
./compare_orchestrator_direct.py --suite thinking --use-baseline

# Create baseline from architect model
./compare_orchestrator_direct.py --create-baseline --suite all

# Tier-balanced sampling (equal questions per difficulty tier)
./compare_orchestrator_direct.py --debug --suite all --stratify-tiers
```

The `--restart-api` flag restarts only the uvicorn API (port 8000), not the llama-server backends (8080-8090). Use after Python code changes.

</details>

<details>
<summary>Code: output format and telemetry</summary>

### Output Format

Per-prompt line includes latency and tokens/sec:
```
  [thinking] t3_q1...   3042ms   16.3 t/s  speedup: 2.1x, quality: OK, turns: 1, routed: frontdoor
```

Per-suite mini-summary (in `run_orchestrator_benchmark.py`):
```
    thinking                10 prompts  ✓  92.0% quality   3042ms avg  16.3 t/s
```

Phase 2 aggregate:
```
  Phase 2 totals: 80 prompts across 8 suites in 342s
    Quality: ✓ 91.2% avg
    Speed:   19.4 t/s avg
    Latency: 4120ms avg
```

### Routing Telemetry

Each response includes `routed_to`, `role_history`, `routing_strategy`, and `tokens_generated` fields for debugging routing decisions.

</details>

</details>

## 3-Way Seeding Infra Hardening (2026-02-08)

To reduce pathological waits on stalled inference ports while preserving reward signal, per-call timeouts are now adaptive by role, mode, and modality. The timeout gets bumped based on observed latency from earlier legs of the same question, so heavyweight architect calls do not get killed by a timeout calibrated for a fast frontdoor response.

<details>
<summary>Timeout and recovery details</summary>

- Per-call timeout is now adaptive by role/mode/modality.
- Timeout is bumped from observed earlier legs for the same question.
- Heavy ports are prechecked before architect calls.
- Zero-token infra on heavy paths gets one recovery retry.
- Tool telemetry fields are normalized together (`tools_used`, `tools_called`, `tool_timings`).

</details>

## 3-Way Live Progress Telemetry (2026-02-09)

Long forced-role calls used to be black boxes until the HTTP response came back. Now the seeding pipeline polls each llama-server's `/slots` endpoint once per second, emitting progress lines so you can see tokens being generated in real time. If a run ends as `INFRA` with zero tokens but the slot counters advanced, the logs surface that discrepancy.

<details>
<summary>Telemetry fields and VL dry-run snapshot</summary>

- Seeding now polls llama-server `/slots` once per second during each forced call.
- Terminal logs emit periodic progress lines:
  - `[slot-progress] <ACTION> port=<PORT> task=<ID> decoded=<N> remain=<M> elapsed=<S>`
- `RoleResult` now carries:
  - `tokens_generated_estimate`
  - `backend_task_id`
  - `slot_progress_source`
- If a run ends as `INFRA` with `tokens_generated=0` but slot counters advanced, logs show:
  - `0 tok, est <N> tok`

This specifically addresses cases where backend generation happened but the orchestrator response closed early and returned no token count.

### VL Dry-run Behavior Snapshot

| Phase | Before (example logs) | After (2026-02-08 dry-run `vl_ocr_0149`) |
|-------|------------------------|-------------------------------------------|
| SELF direct route key | `worker_vision` | `worker_vision:direct` |
| SELF repl route key | `worker_vision:react` (legacy logs) | `worker_vision:repl` |
| ARCHITECT route key | `vision_escalation` | `vision_escalation:direct` |
| Timeout behavior shown | often implicit global timeout | explicit per-call timeout logged (`148s`, `240s`, `212s`) |
| Vision architect token output | mixed in prior runs | generated (`2241 tok` in `159.6s`) |

<details>
<summary>Data: orchestrator results directory</summary>

```
benchmarks/results/orchestrator/
├── comparison_{suite}_{timestamp}.json  # Per-suite comparison
├── run_{timestamp}.json                 # Full run metadata
```

</details>

</details>

## Comparing Models

You can list all previous benchmark runs and compare any two side by side, which is essential for tracking whether a new configuration actually improved things or just shifted the trade-off curve.

<details>
<summary>Code: comparison commands</summary>

```bash
# List all benchmark runs
./scripts/benchmark/compare_results.sh --list-runs

# Compare two runs
./scripts/benchmark/compare_results.sh --baseline RUN_ID --current RUN_ID
```

</details>

## Permanent Results

Benchmark results persist in `benchmarks/results/` even after models are deleted from disk. This matters because storage is finite, new models arrive constantly, and you need historical comparisons to spot trends. Each result includes the full configuration (MoE settings, K values, quantization) so any run can be reproduced later.

<details>
<summary>Design rationale</summary>

This is important because:
- Storage is limited even on our large system
- New models arrive frequently
- Historical comparison enables trend analysis
- Results include configs (MoE settings, K values) for reproduction

</details>

## References

<details>
<summary>Literature and resources</summary>

### LLM Evaluation and Benchmarking

1. Zheng, L., Chiang, W. L., Sheng, Y., Zhuang, S., Wu, Z., Zhuang, Y., ... & Stoica, I. (2023). *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena*. NeurIPS 2023. https://arxiv.org/abs/2306.05685

2. Hendrycks, D., Burns, C., Basart, S., Zou, A., Mazeika, M., Song, D., & Steinhardt, J. (2021). *Measuring Massive Multitask Language Understanding*. ICLR 2021. https://arxiv.org/abs/2009.03300

3. Chen, M., Tworek, J., Jun, H., Yuan, Q., Pinto, H. P. D. O., Kaplan, J., ... & Zaremba, W. (2021). *Evaluating Large Language Models Trained on Code*. arXiv preprint. https://arxiv.org/abs/2107.03374

### Instruction Following and Format Compliance

4. Zhou, J., Lu, T., Mishra, S., Brahma, S., Basu, S., Luan, Y., ... & Hui, K. (2023). *Instruction-Following Evaluation for Large Language Models*. arXiv preprint. https://arxiv.org/abs/2311.07911

5. Ouyang, L., Wu, J., Jiang, X., Almeida, D., Wainwright, C., Mishkin, P., ... & Lowe, R. (2022). *Training Language Models to Follow Instructions with Human Feedback*. NeurIPS 2022. https://arxiv.org/abs/2203.02155

### Long Context Evaluation

6. Liu, N. F., Lin, K., Hewitt, J., Paranjape, A., Bevilacqua, M., Petroni, F., & Liang, P. (2024). *Lost in the Middle: How Language Models Use Long Contexts*. TACL 2024. https://arxiv.org/abs/2307.03172

7. Kamradt, G. (2023). *Needle in a Haystack: Pressure Testing LLMs*. GitHub Repository. https://github.com/gkamradt/LLMTest_NeedleInAHaystack

### LLM-as-Judge Methodology

8. Chiang, W. L., & Zheng, L. (2024). *Chatbot Arena: An Open Platform for Evaluating LLMs by Human Preference*. https://chat.lmsys.org/

9. Dubois, Y., Li, X., Taori, R., Zhang, T., Gulrajani, I., Ba, J., ... & Hashimoto, T. (2024). *AlpacaFarm: A Simulation Framework for Methods that Learn from Human Feedback*. NeurIPS 2023. https://arxiv.org/abs/2305.14387

### Agentic and Tool Use Evaluation

10. Patil, S. G., Zhang, T., Wang, X., & Gonzalez, J. E. (2023). *Gorilla: Large Language Model Connected with Massive APIs*. arXiv preprint. https://arxiv.org/abs/2305.15334

11. Qin, Y., Liang, S., Ye, Y., Zhu, K., Yan, L., Lu, Y., ... & Sun, M. (2024). *ToolLLM: Facilitating Large Language Models to Master 16000+ Real-world APIs*. ICLR 2024. https://arxiv.org/abs/2307.16789

</details>

---

**See Also:** [Master Benchmark Results](../reference/benchmarks/RESULTS.md) — Complete scores and speeds for all 61 models

---

*Previous: [Chapter 20: Session Persistence](20-session-persistence.md)* | *Next: [Chapter 22: Tool Registry & Agent Roles](22-tool-registry.md)*
