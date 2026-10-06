# epyc-orchestrator

Hierarchical multi-model orchestration for **CPU-only local LLM inference** on AMD EPYC. Routes tasks across 20 llama-server ports + 3 infrastructure services with automatic escalation, speculative decoding, KV-cache compression, a learned routing classifier, episodic memory, and a continuously-running autonomous-optimization loop (AutoPilot).

Running on a single AMD EPYC 9655 (96C/192T, 1.13 TB DDR5-5600). No GPU.

---

## 📚 Knowledge Base — Start Here

This repo is the production substrate; the *why* lives in [epyc-root](https://github.com/pestopoppa/epyc-root):

| Index | What's there |
|---|---|
| **[wiki/INDEX.md (epyc-root)](https://github.com/pestopoppa/epyc-root/blob/main/wiki/INDEX.md)** | Compiled topic articles on serving, routing, hardware optimization, research, and related topics |
| **[handoffs/active/master-handoff-index.md (epyc-root)](https://github.com/pestopoppa/epyc-root/blob/main/handoffs/active/master-handoff-index.md)** | Active cross-repo work queue |
| **[research/deep-dives/ (epyc-root)](https://github.com/pestopoppa/epyc-root/tree/main/research/deep-dives)** | Long-form research analyses |
| **[research/intake_index.yaml (epyc-root)](https://github.com/pestopoppa/epyc-root/blob/main/research/intake_index.yaml)** | Triaged papers and repositories with verdicts |
| **In-repo docs** | [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md), [`docs/chapters/INDEX.md`](docs/chapters/INDEX.md) (17 chapters: runtime, REPL, MemRL, escalation, tools, SkillBank), [`scripts/autopilot/program.md`](scripts/autopilot/program.md) |

---

## What It Does

- **Multi-tier routing** with **learned classifier** (98.7% val acc, in shadow mode pending 24-48 h window): routes tasks to the right model — fast workers for simple queries, architects for complex reasoning.
- **Automatic escalation**: failed or timed-out tasks escalate to a more capable tier.
- **Speculative decoding** — draft models accelerate generation (2–4× depending on target).
- **KV-cache compression** — AM (Attention Matching) `POST /slots/{id}?action=compact` endpoint for 5× compression at zero quality cost; EA (Expected Attention) scoring for importance-weighted eviction.
- **Web search** via SearXNG metasearch (local Docker) with DuckDuckGo + Brave + Qwant fallback.
- **ColBERT retrieval** — multi-vector code + document retrieval via NextPLAID; internal markdown KB indexed (409 files / 13.5 K chunks / 861 MiB / 17-min build).
- **Episodic memory (MemRL)** — FAISS-backed session memory with skill tracking; routing weights converged.
- **Sandboxed REPL** — code execution, web fetch, plugin tools.
- **Vision pipeline** — Qwen2.5-VL-7B worker + Qwen3-VL-30B-A3B escalation, OCR, image understanding.
- **AutoPilot** — autonomous optimization loop with 4 species (Seeder, NumericSwarm, PromptForge, StructuralLab), 4D Pareto archive (quality × speed × −cost × reliability), tiered eval tower (T0/T1/T2), safety gate, experiment journal, GEPA evolutionary prompt mutation, Evolution Manager strategy distillation, episodic-memory routing, constrained-creativity stagnation-gated planner, and (since 2026-05-24) full resilience against operator-initiated orchestrator/llama reloads.

---

## Historical Production Snapshot (2026-05-24)

This is a historical stack snapshot. For the current serving posture, see [epyc-root’s inference-serving wiki](https://github.com/pestopoppa/epyc-root/blob/main/wiki/inference-serving.md).

All servers in this snapshot ran on a single AMD EPYC 9655 via the `production-consolidated-v5` llama.cpp branch (Hadamard auto-rotation, flash attention, KV q4_0 K / f16 V, NPS4 NUMA + CCD work distribution, AVX-512BW 8×8 Q8_0 kernel, OMP idle-spin fix).

### LLM Servers in the May 2026 snapshot (20 ports)

| Role | Model | Quant | Port(s) | Notes |
|---|---|---|---|---|
| frontdoor / worker_summarize | Qwen3.6-35B-A3B-MTP | Q8_0 (37 GB) | 8070, 8080, 8180 | Shared GGUF mmap. `enable_thinking=False` mandatory. 1 full (`-np 16`) + 2 halves (`-np 4`). |
| worker_general / worker_explore / worker_math / toolrunner | gemma-4-26B-A4B-it | Q4_K_M (16 GB) | 8072, 8082, 8182 | Try-cheap-first worker. MTP. Needs `KMP_BLOCKTIME=10` (OMP idle-spin fix). |
| architect_general | Qwen3.8-Flash-Next | UD-IQ4_XS (~87 GiB) | 8074 | CPU, one full 0-95 instance holding all four region locks (serial). Terminal escalation rung. Served `architect_critic` until the 2026-09-27 ARCHITECT SWAP. |
| architect_critic / coder_escalation / ingest_long_context | Qwen3.8-27B | Q8_0 (27 GB) | 8083 | **GPU (MI210)**. One process, three names; hosts `critique_plan`. Served `architect_general` until the 2026-09-27 ARCHITECT SWAP. |
| worker_vision / vision_escalation | Qwen3-VL-30B-A3B-Instruct | Q4_K_M (17 GB) | 8086 | **GPU (MI210)**. One process, both names; `:8087` retired. Needs `max_tokens >= 1024`. |
| embedder pool ×6 | BGE-large-en-v1.5 | f16 (0.6 GB) | 8090–8095 | 1024-dim embeddings. |

**Note:** `architect_coding` (formerly REAP-246B) was retired 2026-05-09 — Qwen3.6-35B-Q8 on coder_escalation beats it by ≈27 pp at <1/7 the memory. See [project_stack_consolidation_2026_05](../epyc-root/wiki/inference-serving.md) for the deconsolidation rationale.

### Infrastructure Services (Docker)

| Service | Port | Purpose |
|---|---|---|
| nextplaid-code | 8088 | ColBERT multi-vector code retrieval (LateOn-Code) |
| nextplaid-docs | 8089 | ColBERT multi-vector doc retrieval (GTE-ModernColBERT) |
| searxng | local | Metasearch (JSON API; DuckDuckGo + Brave + Qwant + Wikipedia consensus) |

---

## AutoPilot: Continuous Optimization

The orchestrator includes an autonomous optimization loop (AutoPilot) that continuously improves prompts, routing, and model configurations through controlled experiments with safety gates.

**513 trials completed** (96.3% trustworthy — 13 historically corrupted by a known chat-template bug, scrubbed). 4 optimizer species, tiered eval tower (T0 = 10 q / 30 s; T1 = 100 q / 5 min; T2 = 500+ q / 30 min), 4D Pareto archive, safety gate with MAD-based statistical noise filtering, atomic experiment journal with planner-trustworthiness gating, full crash-resilience (operator reloads no longer pollute; autopilot SIGKILL is recoverable via WAL-style in-flight marker).

### Diagnostic Plots

The plots committed below are a **2026-04-15 snapshot at trial 192** — captured before the May 2026 stack consolidation and the constrained-creativity / exogenous-restart upgrades. Refreshing them past the latest model swaps is queued under [readme-refresh.md](https://github.com/pestopoppa/epyc-root/blob/main/handoffs/active/readme-refresh.md).

#### Objectives Overview
![Objectives Overview](docs/autopilot/objectives_2x2.png)
Quality (Claude-as-Judge 0–3), speed (e2e t/s), cost (normalized 0–1), reliability (success fraction).

#### Pareto Frontier
![Pareto Frontier](docs/autopilot/pareto_frontier_2d.png)
Quality × speed. End-to-end pipeline speeds (47–65 t/s) exceed any single model because of multi-instance NUMA fan-out, draft acceleration, and REPL amortization.

#### Hypervolume Trend
![Hypervolume Trend](docs/autopilot/hypervolume_trend.png)
Hypervolume indicator over trial count — measures Pareto-dominated volume in objective space.

#### Species Effectiveness
![Species Effectiveness](docs/autopilot/species_effectiveness.png)
Pareto-improvement rate per optimization species.

#### Per-Suite Quality
![Per-Suite Quality](docs/autopilot/per_suite_quality.png)
Quality breakdown by benchmark suite for the current best configuration.

#### Trial Timeline
![Trial Timeline](docs/autopilot/trial_timeline.png)
Chronological view colored by species.

#### Memory Convergence
![Memory Convergence](docs/autopilot/memory_convergence.png)
Q-value TD-error magnitude for MemRL routing — converged (MA-10 ≈ 0, below threshold).

---

## Quick Start

```bash
# 1. Clone
git clone https://github.com/pestopoppa/epyc-orchestrator.git
cd epyc-orchestrator
pip install -e ".[dev]"

# 2. Launch full stack (orchestrator + all llama-servers)
python scripts/server/orchestrator_stack.py start

# 3. Verify live state
curl http://localhost:8000/dashboard/api/version       # orchestrator git_sha + start time
curl http://localhost:8000/dashboard/api/llama_fleet_ids   # per-port marker map
curl http://localhost:8000/health                       # backend probes + knowledge-tool status

# 4. Pre-flight diagnostic + AutoPilot
python scripts/autopilot/preflight_audit.py
python scripts/autopilot/autopilot.py start --tui
```

If every PromptForge mutation is being rejected with `eval_leakage_vocabulary_unavailable`, follow the runbook in [`docs/guides/meta-harness-operator-guide.md` § 7](docs/guides/meta-harness-operator-guide.md#7-mutations-all-rejected-eval_leakage_vocabulary_unavailable).

Dashboard at `http://localhost:8000/dashboard/`.

---

## Architecture

```
Request → FastAPI (:8000) → ChatPipeline → Mode selection
                                            ├── Direct  → LLM call → Response
                                            ├── REPL    → Tool loop → Response
                                            └── Delegated → Architect plan → Worker execution

Model stack (NPS4; 1 full + 2 half instances per CPU role — quarters retired):
  Tier A: frontdoor + architect_general (Qwen3.6-35B-A3B Q8 CPU; Qwen3.8-Flash-Next UD-IQ4_XS, full CPU :8074)
  Tier B: architect_critic (Qwen3.8-27B Q8 on MI210 :8083; hosts coder_escalation + ingest_long_context)
  (2026-09-27 ARCHITECT SWAP: the two architect labels swapped over unchanged processes.)
  Tier C: Workers (gemma-4-26B-A4B MTP ×3 instances), vision (Qwen3-VL-30B on MI210)
  Tier D: Embedders (6× BGE-large)
  Speech: whisper.cpp STT :9000 + qwentts TTS :9002, both on MI210 (live 2026-08-02)

Routing: LearnedRoutingClassifier (98.7%, shadow mode) → falls through to MemRL
         → falls through to rule-based difficulty heuristics

AutoPilot: Controller → 4 species (Seeder / NumericSwarm / PromptForge / StructuralLab)
           → EvalTower (T0 sentinel → T1 deep → T2 full)
           → SafetyGate (MAD-noise-filtered) → ParetoArchive → Journal
           → Evolution Manager (strategy distillation every 5 trials)
           → OrchestratorWatcher + resilient_post (exogenous-reload resilience)
```

---

## Eval Suites

Benchmarks live in the [epyc-inference-research](https://github.com/pestopoppa/epyc-inference-research) repo (30+ suites, 57 K questions). The autopilot eval tower references them via shared model registry. See:

- [Master results table](https://github.com/pestopoppa/epyc-inference-research/blob/main/docs/reference/benchmarks/RESULTS.md)
- [Benchmark methodology wiki](https://github.com/pestopoppa/epyc-root/blob/main/wiki/benchmark-methodology.md)

---

## Documentation

| Doc | What's there |
|---|---|
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | Module responsibilities, request flow |
| [`docs/chapters/INDEX.md`](docs/chapters/INDEX.md) | 17 deep chapters: runtime, REPL, MemRL, escalation, tools, SkillBank |
| [`scripts/autopilot/program.md`](scripts/autopilot/program.md) | Operator-editable strategy doc the controller reads |
| [`scripts/server/orchestrator_stack.py`](scripts/server/orchestrator_stack.py) | Single source of truth for the stack launch (HOT/WARM, NUMA pinning, OMP env) |
| [`orchestration/model_registry.yaml`](orchestration/model_registry.yaml) | Active stack + drafter catalogue (~80 entries) |

---

## Development

```bash
pytest tests/ -n 8                             # parallel test run
ruff check src/                                # lint
python scripts/autopilot/preflight_audit.py    # 9-check diagnostic
```

Pre-edit gate for non-trivial changes: run `gitnexus impact <symbol> --direction upstream` only when the selected index is trusted and current (see [CLAUDE.md](CLAUDE.md)). The canonical host index is currently untrusted; this advice does not authorize querying or recovering it ([incident record](https://github.com/pestopoppa/epyc-root/blob/main/docs/reference/agent-config/INCIDENT_LOG.md#inc-20261005-unclaimed-host-code-index)). The wrapper script `scripts/gitnexus-analyze.sh` requires eight live CPU-region claims before invoking GitNexus and serializes writers; see the ROOT wiki’s [NI76 implementation record](https://github.com/pestopoppa/epyc-root/blob/main/wiki/tool-implementation.md).

---

## License

MIT — see [LICENSE](LICENSE). Model licenses vary; see registry entries.
