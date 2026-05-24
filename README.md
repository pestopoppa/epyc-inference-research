# epyc-inference-research

CPU inference optimization research, benchmarks, and model evaluation for AMD EPYC 9655. Houses the 57 K-question pool, 30+ eval suites, master results table, model registry, and per-thread experimental scripts that power the autopilot optimization loop in [epyc-orchestrator](https://github.com/pestopoppa/epyc-orchestrator).

Single AMD EPYC 9655 "Turin" — 96C/192T (Zen 5), 1.13 TB DDR5-5600 ECC across 12 channels (~460 GB/s aggregate), NPS4 NUMA. **CPU-only**; no GPU.

---

## 📚 Knowledge Base — Start Here

The "why" behind every benchmark, model swap, and methodology decision lives in [epyc-root](https://github.com/pestopoppa/epyc-root):

| Index | What's there |
|---|---|
| **[wiki/INDEX.md (epyc-root)](https://github.com/pestopoppa/epyc-root/blob/main/wiki/INDEX.md)** | 30 compiled topic articles — benchmark methodology, speculative decoding, KV cache, MoE, NUMA, quantization, … |
| **[wiki/benchmark-methodology.md](https://github.com/pestopoppa/epyc-root/blob/main/wiki/benchmark-methodology.md)** | Compiled methodology synthesis across all benchmark-related sources |
| **[research/deep-dives/ (epyc-root)](https://github.com/pestopoppa/epyc-root/tree/main/research/deep-dives)** | 105 long-form analyses of individual papers / techniques |
| **[research/intake_index.yaml (epyc-root)](https://github.com/pestopoppa/epyc-root/blob/main/research/intake_index.yaml)** | 595 triaged papers/repos with credibility scores + verdicts |
| **In-repo docs** | [`docs/chapters/`](docs/chapters/) (10 inference-optimization chapters), [`docs/guides/`](docs/guides/) (benchmarking, KV compaction, model sizing), [`docs/reference/benchmarks/RESULTS.md`](docs/reference/benchmarks/RESULTS.md) (master results table) |

---

## In-Repo Reference

| Doc | What's there |
|---|---|
| **[`docs/chapters/INDEX.md`](docs/chapters/INDEX.md)** | 10 inference-optimization chapters: speculative decoding, MoE optimization, prompt lookup, radix attention, deprecated approaches, benchmarking framework, benchmark suite construction, cost-aware rewards, claude debugger, advanced speculative decoding |
| **[`docs/guides/benchmarking-guide.md`](docs/guides/benchmarking-guide.md)** | End-to-end benchmark workflow |
| **[`docs/guides/kv-compaction-guide.md`](docs/guides/kv-compaction-guide.md)** | Attention-matching KV-cache compaction recipe |
| **[`docs/guides/model-sizing.md`](docs/guides/model-sizing.md)** | Model size vs. memory/throughput tradeoffs |
| **[`docs/reference/benchmarks/RESULTS.md`](docs/reference/benchmarks/RESULTS.md)** | Master results table — every benchmark run |
| **[`docs/reference/models/QUIRKS.md`](docs/reference/models/QUIRKS.md)** | Known model issues + workarounds |
| **[`docs/MODEL_MANIFEST.md`](docs/MODEL_MANIFEST.md)** | Per-model lineage + provenance |
| **[`research/`](research/)** | Per-thread experimental plans + investigations (agentic / coder / formalizer / escalation flow / hierarchical orchestration / K-value sweeps / ...) |
| **[`orchestration/model_registry.yaml`](orchestration/model_registry.yaml)** | Comprehensive benchmark-record registry (broader than the orchestrator's active stack) |

---

## Eval Infrastructure

**57 K+ questions across 30+ suites** with automated deterministic scoring (exact_match / substring / multiple_choice / f1 / code_execution / llm_judge).

| Category | Suites | Questions | Scoring |
|---|---|---:|---|
| General knowledge | MMLU, SimpleQA, HotpotQA | 25 K+ | multiple_choice, f1, substring |
| Math / reasoning | GSM8K, AIME, OlympiadBench, MATH-500 | 3 K+ | exact_match, substring |
| Code | MBPP, BigCodeBench, LiveCodeBench, CRUXEval, USACO | 6 K+ | substring, code_execution |
| Science | GPQA, PHYBench, PhysReason | 3.6 K | multiple_choice, llm_judge |
| Long context | ZeroSCROLLS, LEval, LongBench, RULER, Needle | 1.6 K | llm_judge, exact_match |
| Reasoning | HellaSwag, DebugBench | 15 K+ | multiple_choice |
| Vision | OCRBench (VL) | 2.5 K | exact_match |
| Tool use | Agentic, Web Research, Skill Transfer | 130 | f1, exact_match |
| Hard | Mode Advantage, Mode Advantage Hard | 150 | substring |

The active 39-question sentinel pool spans GPQA, olympiad math, multi-hop QA, tool use, and structured extraction — selected for diversity + speed (T0 in 30 s, T1 in 5 min).

---

## Recent Results (last 60 days)

| Date | Result | Where to read |
|---|---|---|
| 2026-05-08 | **gemma-4-26B-A4B MTP** beats Qwen3-Coder-30B as worker_general: +18 pp tool_compliance, +6 pp full suite, +36% throughput (76.5 t/s solo) via ik_llama.cpp PR #1744 | [wiki/moe-optimization.md](https://github.com/pestopoppa/epyc-root/blob/main/wiki/moe-optimization.md) |
| 2026-05-06 | **Qwen3.6-35B-A3B Q8** drop-in upgrade for frontdoor: +33 pp accuracy + 80% t/s vs. prior Qwen3.5-35B-A3B Q4_K_M (mandatory `enable_thinking=False`) | [wiki/inference-serving.md](https://github.com/pestopoppa/epyc-root/blob/main/wiki/inference-serving.md) |
| 2026-04-24 | **NPS4 + CCD work distribution + AVX-512BW 8×8 Q8_0 kernel** — best single-instance config is 48-thread @ 46.6 t/s for 30B-A3B Q4_K_M (+15% vs 96t); single-thread +31.8% | [wiki/hardware-optimization.md](https://github.com/pestopoppa/epyc-root/blob/main/wiki/hardware-optimization.md) |
| 2026-04-26 | **L3-as-NUMA reverted** — all 5 production models −30 to −52% on L3aaN BIOS layout; NPS4 is the correct topology | [wiki/hardware-optimization.md](https://github.com/pestopoppa/epyc-root/blob/main/wiki/hardware-optimization.md) |
| 2026-04 ongoing | **Attention Matching KV compaction** — 50× compression at zero PPL cost; native `/slots/{id}?action=compact` endpoint; autopilot slot_compact integration complete | [wiki/kv-cache.md](https://github.com/pestopoppa/epyc-root/blob/main/wiki/kv-cache.md), [`docs/guides/kv-compaction-guide.md`](docs/guides/kv-compaction-guide.md) |

The full headline-throughput map per quant per role lives in [`docs/reference/benchmarks/RESULTS.md`](docs/reference/benchmarks/RESULTS.md).

---

## Running Benchmarks

```bash
# Build/refresh the question pool
python scripts/benchmark/question_pool.py --build

# 3-way routing evaluation (frontdoor vs coder vs worker)
python scripts/benchmark/seed_specialist_routing.py \
    --3way --suites math coder general --sample-size 20 --tui

# Specific suite, specific model
python scripts/benchmark/run_benchmark.py \
    --suite gpqa --model frontdoor --sample-size 50
```

**Methodology guard rails** (per [feedback memories](https://github.com/pestopoppa/epyc-root/blob/main/wiki/benchmark-methodology.md)):

- Index results by **model + quant + flags**, never by orchestrator role (roles get reassigned; reassignment must not lose data).
- Always run a sweep — never deploy without measured numbers.
- `llama-bench` defaults to `-fa 0`; **always pass `-fa 1` explicitly** for CPU decode (~8–10% swing).
- Never pipe `llama-cli` output through `grep`/`tail`/`head` — redirect to file then `cat`.
- Single-model vs NUMA-concurrent modes need **independently optimized** params; don't reuse settings across regimes.

---

## Repository Layout

```
epyc-inference-research/
├── README.md                              # this file
├── docs/
│   ├── MODEL_MANIFEST.md
│   ├── chapters/                          # 10 inference-optimization chapters
│   ├── guides/                            # benchmarking-guide, kv-compaction-guide, model-sizing
│   ├── reference/
│   │   ├── benchmarks/RESULTS.md          # master results table
│   │   ├── benchmarks/SERVER_MODE.md
│   │   └── models/QUIRKS.md
│   └── experiments/
│
├── research/                              # per-thread experimental plans
│   ├── AGENTIC_BENCHMARKING_PLAN.md
│   ├── CODER_BENCHMARKING_PLAN.md
│   ├── GENERAL_BENCHMARKING_PLAN.md
│   ├── INSTRUCTION_PRECISION_BENCHMARKING_PLAN.md
│   ├── ESCALATION_FLOW.md
│   ├── FORMALIZER_INVESTIGATION.md
│   ├── Hierarchical_Orchestration_Methodology.md
│   ├── K_VALUE_OPTIMIZATION.md
│   ├── CAS_SPEC_IMPLEMENTATION_PLAN.md
│   └── ... (per-thread plans + investigations)
│
├── scripts/
│   ├── benchmark/                         # 60+ scripts: question_pool, seed_*, bench_*, analyze_*
│   ├── server/                            # llama-server lifecycle helpers
│   └── utils/
│
├── orchestration/
│   ├── model_registry.yaml                # comprehensive benchmark-record registry
│   ├── optimization_checkpoint.yaml
│   ├── optimization_report.md
│   └── optuna_study.db                    # NumericSwarm hyperparameter search DB
│
├── benchmarks/                            # per-run output JSONs
├── configs/                               # benchmark config templates
└── data/                                  # question pool + scoring fixtures
```

---

## Cross-Repo Companions

- **[epyc-orchestrator](https://github.com/pestopoppa/epyc-orchestrator)** — production orchestration that consumes these benchmarks via the autopilot eval tower
- **[epyc-root](https://github.com/pestopoppa/epyc-root)** — governance + compiled knowledge base
- **[llama.cpp fork](https://github.com/pestopoppa/llama.cpp)** — `production-consolidated-v5` (Hadamard auto-rotation, KV q4_0, NPS4-aware, AVX-512BW Q8 kernel, OMP idle-spin fix)

---

## License

MIT. Models are under their own licenses; see per-entry notes in [intake_index.yaml](https://github.com/pestopoppa/epyc-root/blob/main/research/intake_index.yaml).
