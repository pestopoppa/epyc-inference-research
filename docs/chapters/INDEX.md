# Research Chapters Index

29 chapters documenting AMD EPYC 9655 inference optimization, orchestration architecture, and intelligence systems.

## Reading Paths

### For Contributors (New to the project)

1. [Ch01](01-hardware-system.md) Hardware -> [Ch05](05-speculative-decoding.md) Spec Decode -> [Ch06](06-moe-optimization.md) MoE -> [Ch07](07-prompt-lookup.md) Prompt Lookup
2. [Ch03](03-llama-cpp-toolchain.md) Toolchain -> [Ch09](09-deprecated-approaches.md) Dead Ends -> [Ch21](21-benchmarking-framework.md) Benchmarking
3. [Ch10](10-orchestration-architecture.md) Architecture -> [Ch12](12-production-server-stack.md) Server Stack -> [Ch22](22-tool-registry.md) Tools

### For Agents / Daily Reference

| Task | Start Here |
|------|-----------|
| Which model for X? | [Ch10](10-orchestration-architecture.md) + [MODELS.md](../reference/models/MODELS.md) |
| Debugging escalation | [Ch18](18-escalation-and-routing.md) + [Ch16](16-graph-reasoning.md) |
| Server won't start | [Ch12](12-production-server-stack.md) + [Ch04](04-storage-and-safety.md) |
| Adding a tool | [Ch22](22-tool-registry.md) + [Ch11](11-repl-environment.md) |
| Benchmarking a model | [Ch21](21-benchmarking-framework.md) + [Ch24](24-benchmark-suite-construction.md) |
| Memory/learning system | [Ch15](15-memrl-system.md) + [Ch16](16-graph-reasoning.md) + [Ch17](17-memory-seeding.md) + [Ch25](25-cost-aware-rewards.md) + [Ch27](27-skillbank-experience-distillation.md) + [Ch28](28-calibration-and-risk-control.md) |

### For Researchers / Public Showcase

Novel contributions (recommended reading order):

1. [Ch05](05-speculative-decoding.md) 11x CPU speedup via speculative decoding
2. [Ch06](06-moe-optimization.md) Quality-preserving expert reduction
3. [Ch15](15-memrl-system.md) MemRL: reinforcement learning for model routing
4. [Ch16](16-graph-reasoning.md) Graph-based failure and hypothesis reasoning
5. [Ch10](10-orchestration-architecture.md) Hierarchical 4-tier agent architecture
6. [Ch21](21-benchmarking-framework.md) 8-suite framework with Claude-as-Judge
7. [Ch27](27-skillbank-experience-distillation.md) SkillBank: experience distillation for routing skill transfer

---

## Complete Chapter List

### Part I: Foundation

| # | Title | Key Result |
|---|-------|------------|
| 01 | [Hardware System](01-hardware-system.md) | ~460 GB/s bandwidth, 1.13TB RAM |
| 02 | [Runtime Environment](02-runtime-environment.md) | 10 feature flags, centralized config |
| 03 | [llama.cpp Toolchain](03-llama-cpp-toolchain.md) | 3 upstream PRs, production branch safety |
| 04 | [Storage & Safety](04-storage-and-safety.md) | RAID0 rules, 192-thread memory safety |

### Part II: Inference Optimization

| # | Title | Key Result |
|---|-------|------------|
| 05 | [Speculative Decoding](05-speculative-decoding.md) | **11x** speedup on code generation |
| 06 | [MoE Expert Reduction](06-moe-optimization.md) | **+52%** on 30B MoE models |
| 07 | [Prompt Lookup](07-prompt-lookup.md) | **12.7x** on summarization tasks |
| 08 | [RadixAttention](08-radix-attention.md) | >50% cache hit on orchestrator |
| 09 | [Deprecated Approaches](09-deprecated-approaches.md) | EAGLE-1, CAS-Spec lessons |

### Part III: System Architecture

| # | Title | Key Result |
|---|-------|------------|
| 10 | [Orchestration Architecture](10-orchestration-architecture.md) | 4-tier agents, TaskIR, escalation |
| 11 | [REPL Environment](11-repl-environment.md) | AST sandbox, Research Context Tracker |
| 12 | [Production Server Stack](12-production-server-stack.md) | 9 servers, ~535GB HOT tier |
| 13 | [Data Processing Pipelines](13-data-processing-pipelines.md) | 19x OCR speedup, vision batch |
| 14 | [TOON Encoding](14-toon-encoding.md) | 55% token compression |

### Part IV: Intelligence & Learning

| # | Title | Key Result |
|---|-------|------------|
| 15 | [MemRL System](15-memrl-system.md) | 2,714 memories (cleaned), FAISS, Q-scoring |
| 16 | [Graph-Based Reasoning](16-graph-reasoning.md) | 13 failure modes, Kuzu backend |
| 17 | [Memory Seeding](17-memory-seeding.md) | 56 seeds, 10 strategies incl 3-way eval |
| 18 | [Escalation & Routing](18-escalation-and-routing.md) | 3-way confidence routing, general delegation |
| 19 | [Procedure Registry](19-procedure-registry.md) | 11 procedures, ~350 tokens/op |
| 20 | [Session Persistence](20-session-persistence.md) | 7-phase checkpoint/resume |

### Part V: Operations & Quality

| # | Title | Key Result |
|---|-------|------------|
| 21 | [Benchmarking Framework](21-benchmarking-framework.md) | 8 suites, 77 models scored |
| 22 | [Tool Registry & Agent Roles](22-tool-registry.md) | 40+ tools, 8 agent roles |
| 23 | [Security & Monitoring](23-security-and-monitoring.md) | AST sandbox, entropy detection |
| 24 | [Benchmark Suite Construction](24-benchmark-suite-construction.md) | 325 questions, 5 scoring methods |
| 25 | [Cost-Aware Rewards](25-cost-aware-rewards.md) | Binary rewards for P(success), cost stored separately |
| 26 | [Claude-in-the-Loop Debugger](26-claude-debugger.md) | 12 anomaly signals, hot-swap fixes, 3-phase regression |
| 27 | [SkillBank & Experience Distillation](27-skillbank-experience-distillation.md) | Structured skill library, teacher distillation, recursive evolution |
| 28 | [Calibration and Risk Control](28-calibration-and-risk-control.md) | ECE/Brier/coverage/risk metrics + conformal thresholding |
| 29 | [Programmatic Tool Chaining](29-programmatic-tool-chaining.md) | Deferred tool results, safe chaining, cross-request REPL persistence |

---

## Cross-Reference

- **Living Technical Reference**: [ARCHITECTURE.md](../ARCHITECTURE.md) (updated continuously)
- **Benchmark Data**: [RESULTS.md](../reference/benchmarks/RESULTS.md)
- **Model Registry**: [model_registry.yaml](../../orchestration/model_registry.yaml)
- **Model Quirks**: [QUIRKS.md](../reference/models/QUIRKS.md)
- [Back to README](../../README.md)
