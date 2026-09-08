# epyc-inference-research

CPU inference optimization research, benchmarks, and model evaluation for AMD EPYC 9655. Houses the ~79 K-question pool, 38 eval suites, master results table, model registry, and per-thread experimental scripts that power the autopilot optimization loop in [epyc-orchestrator](https://github.com/pestopoppa/epyc-orchestrator).

Single AMD EPYC 9655 "Turin" — 96C/192T (Zen 5), 1.13 TB DDR5-5600 ECC across 12 channels (~460 GB/s aggregate), NPS4 NUMA. CPU is the primary inference regime; an AMD Instinct MI210 (gfx90a, 64 GB, ROCm) is also present and is used as a second measurement/serving lane — GPU rows in `docs/data/` and `orchestration/model_registry.yaml` are labelled by device.

---

## 📚 Knowledge Base — Start Here

The "why" behind every benchmark, model swap, and methodology decision lives in [epyc-root](https://github.com/pestopoppa/epyc-root):

| Index | What's there |
|---|---|
| **[wiki/INDEX.md (epyc-root)](https://github.com/pestopoppa/epyc-root/blob/main/wiki/INDEX.md)** | 29 compiled topic articles — benchmark methodology, speculative decoding, KV cache, MoE, NUMA, quantization, SSM-hybrid, … |
| **[wiki/benchmark-methodology.md](https://github.com/pestopoppa/epyc-root/blob/main/wiki/benchmark-methodology.md)** | Compiled methodology synthesis across all benchmark-related sources |
| **[research/deep-dives/ (epyc-root)](https://github.com/pestopoppa/epyc-root/tree/main/research/deep-dives)** | 137 long-form analyses of individual papers / techniques |
| **[research/intake_index.yaml (epyc-root)](https://github.com/pestopoppa/epyc-root/blob/main/research/intake_index.yaml)** | 936 triaged papers/repos with credibility scores + verdicts |
| **[MEASUREMENT.md (epyc-root)](https://github.com/pestopoppa/epyc-root/blob/main/MEASUREMENT.md)** | Instrument constitution — what makes a number decision-grade vs. an observation |
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
| **[`docs/reference/models/MODELS.md`](docs/reference/models/MODELS.md)** | Model catalogue reference |
| **[`docs/reference/GENERATED_DOCS_INDEX.md`](docs/reference/GENERATED_DOCS_INDEX.md)** · **[`docs/reference/ANALYSIS_REPORTS_INDEX.md`](docs/reference/ANALYSIS_REPORTS_INDEX.md)** | Generated indices over every doc / analysis artifact in the repo |
| **[`docs/reference/model-probe-scoreboard.md`](docs/reference/model-probe-scoreboard.md)** | Pointer + row schema; the canonical scoreboard lives in epyc-root (do not fork it here) |
| **[`scripts/kernel_rnd/autokernel/controller/ARENA_INTEGRATION.md`](scripts/kernel_rnd/autokernel/controller/ARENA_INTEGRATION.md)** | Governed AgentKernelArena controller and raw-HIP authoring seams for MI210 |
| **[`docs/MODEL_MANIFEST.md`](docs/MODEL_MANIFEST.md)** | Per-model lineage + provenance (live-deployment truth is compiled in epyc-orchestrator) |
| **[`docs/data/`](docs/data/)** | Dated evidence packets + measurement manifests for individual campaigns |
| **[`docs/design/`](docs/design/)** | Measurement/harness design specs written before anything is run |
| **[`docs/experiments/`](docs/experiments/)** | Per-experiment analyses (KV compaction, self-speculation, TrimR, SEAL, HiSpec, …) |
| **[`research/`](research/)** | Per-thread experimental plans + investigations (agentic / coder / formalizer / escalation flow / hierarchical orchestration / K-value sweeps / ...) |
| **[`handoffs/active/master-handoff-index.md`](handoffs/active/master-handoff-index.md)** | Outstanding research-repo work items (production/stack authority stays in epyc-root + epyc-orchestrator) |
| **[`orchestration/model_registry.yaml`](orchestration/model_registry.yaml)** | Comprehensive benchmark-record registry — 179 role/candidate entries, broader than the orchestrator's active stack |

---

## Eval Infrastructure

**79,479 questions across 38 non-empty suites** (pool built 2026-07-27 — regenerate with `question_pool.py --build`, inspect with `--stats`) and nine automated scoring methods: `exact_match`, `structural_exact_match`, `substring`, `multiple_choice`, `f1`, `f1_list`, `code_execution`, `programmatic`, `llm_judge`.

| Category | Suites | Questions | Scoring |
|---|---|---:|---|
| General knowledge | `general`, `mmlu_pro`, `hotpotqa`, `simpleqa` | 37,805 | multiple_choice, f1 |
| Thinking / long CoT | `thinking`, `longcot_mini` | 11,616 | multiple_choice, structural_exact_match |
| Code | `debugbench`, `livecodebench`, `cruxeval`, `bigcodebench`, `coder`, `usaco` | 10,537 | code_execution, substring, exact_match |
| Scoring verifiers & memory | `scoring_verifiers`, `omniscience`, `instruction_precision`, `tulving_episodic`, `real_suite_v1` | 8,348 | multiple_choice, f1, f1_list, programmatic |
| Science | `physreason`, `gpqa`, `gpqa_diamond`, `gpqa_diamond_cot`, `physics` | 4,061 | llm_judge, multiple_choice |
| Math | `math`, `olympiadbench`, `aime`, `aime25` | 2,583 | exact_match, substring |
| Vision | `vl` | 2,575 | substring, exact_match |
| Long context | `zeroscrolls`, `leval`, `longbench`, `ruler`, `needle_parameterized`, `long_context` | 1,674 | llm_judge, exact_match, substring |
| Hard | `mode_advantage`, `mode_advantage_hard` | 150 | exact_match, substring, code_execution |
| Tool use | `agentic`, `web_research`, `skill_transfer` | 130 | substring, f1 |

**Code-execution scoring** runs through [`scripts/benchmark/code_exec_scorer.py`](scripts/benchmark/code_exec_scorer.py): fresh temp cwd, `RLIMIT_CPU` / `RLIMIT_AS` / `RLIMIT_CORE=0`, wall-clock timeout, minimal env, and a **dedicated process group** so a timeout kills the child's whole descendant tree rather than leaving orphans behind. Process/thread *count* is deliberately unbounded — `RLIMIT_NPROC` is enforced per real UID rather than per process tree, and this host runs ~9.5 K threads under one uid, so any per-scorer cap would fail the child's first fork nondeterministically under fleet load; cgroup v2 `pids.max` is the tracked correct mechanism. The scorer provides **no** network isolation and no filesystem jail — point it at trusted algorithmic benchmarks only.

**Multi-turn agentic SWE evaluation** runs through [`scripts/benchmark/agentic_swe_harness.py`](scripts/benchmark/agentic_swe_harness.py) — a no-oracle repo-fixing loop (`bash` / `edit` / `done` actions) against a SWE-bench instance container, emitting `swebench.harness.run_evaluation`-compatible prediction rows. It calls `DockerEnv.reset_testbed(base_commit)` **fail-closed before every trial**: a failed reset produces status `testbed_reset_failed` instead of a scored row, so repeated-trial sweeps cannot score against the previous trial's leftovers. Results are labelled by arm (model/quant), never by role.

The active 39-question sentinel pool spans GPQA, olympiad math, multi-hop QA, tool use, and structured extraction — selected for diversity + speed (T0 in 30 s, T1 in 5 min).

---

## Recent Results (last 60 days)

Per [MEASUREMENT.md](https://github.com/pestopoppa/epyc-root/blob/main/MEASUREMENT.md), a number without a protocol citation is an **observation**, not a decision-gating result. The rows below name *what changed* and point at the artifact that carries the protocol, identity witnesses and grade — read the numbers there, not here.

| Date | Result | Where to read |
|---|---|---|
| 2026-07-29 | **Scoring-infra hardening.** `code_exec_scorer.py` now runs the scored subprocess in its own process group and SIGKILLs the whole group on timeout (no more surviving orphans); `agentic_swe_harness.py` gained a fail-closed `/testbed` reset before every trial, closing a hole where repeated-trial sweeps scored against the previous trial's leftovers | [`scripts/benchmark/code_exec_scorer.py`](scripts/benchmark/code_exec_scorer.py), [`scripts/benchmark/agentic_swe_harness.py`](scripts/benchmark/agentic_swe_harness.py) |
| 2026-07-25 | **Production kernel v8 freeze.** `/mnt/raid0/llm/llama.cpp` is frozen on `production-consolidated-v8` at `67a433bf45a8a091d83b4ea0b32ff0735fd51800` (`llama-server --version` → `10107`), a single-kernel serving path — the separate `ik_llama.cpp` binary is fully deprecated as a serving path | [epyc-root CLAUDE.md](https://github.com/pestopoppa/epyc-root/blob/main/CLAUDE.md), [`docs/data/laguna_iq2_mi210_kv_sweep_20260725.md`](docs/data/laguna_iq2_mi210_kv_sweep_20260725.md) |
| 2026-07-25 | **Laguna-S-2.1 UD-IQ2_M K/V + flash-attention sweep on MI210** — exact-tip run with full pre/post source, server, shared-library, model, harness and execution-binding witnesses. Explicitly labelled *observation only*: not a promotion gate and not a global-optimum claim | [`docs/data/laguna_iq2_mi210_kv_sweep_20260725.md`](docs/data/laguna_iq2_mi210_kv_sweep_20260725.md) |
| 2026-07-18 → 07-20 | **CPU prefill-compute trace campaign (PC-0 → PC-4m).** A ~15-step profiling chain through the MoE/FFN/router/scheduler path, ending in an opt-in `GGML_CPU_CONCAT_DIM0_ROWS=1` CPU fast path in `llama.cpp-experimental` with a real support predicate. Post-candidate research — the frozen v7 promotion candidate `6ad45fa3ff` was not updated by it | [`docs/data/cpu_prefill_compute_pc4m_concat_dim0_hardening_20260720.md`](docs/data/cpu_prefill_compute_pc4m_concat_dim0_hardening_20260720.md) and the `docs/data/cpu_prefill_compute_pc*` series |
| 2026-07-18 | **GLM-MTP / sparse final-attention prep** (zero-inference source prep). No upstream path supports complete GLM/GLM-DSA native MTP or real indexed sparse DSA final attention; the experimental tree has a buildable GLM-5.2 `glm-dsa` single-NextN scaffold only. No throughput or quality claims until the GLM reviewer gate closes | [`docs/reference/benchmarks/glm_mtp_sparse_attention_prep_20260718.md`](docs/reference/benchmarks/glm_mtp_sparse_attention_prep_20260718.md), [`docs/reference/kernel/glm-mtp-sparse-attention-implementation-map-20260718.md`](docs/reference/kernel/glm-mtp-sparse-attention-implementation-map-20260718.md) |
| 2026-07-16 → 07-20 | **Model-admission checkpoint + artifact audit** for the quiet-window backlog (GLM-5.2, Hy3, Bonsai / Ternary Bonsai, Qwable, Nemotron, Gemma4). Research candidates only — nothing here is promoted into the lean orchestrator registry without an explicit stack-change handoff | [`docs/reference/models/model-admission-2026-07-16.md`](docs/reference/models/model-admission-2026-07-16.md) |

The full headline-throughput map per quant per role lives in [`docs/reference/benchmarks/RESULTS.md`](docs/reference/benchmarks/RESULTS.md); the standing rationale for the current model lineup is in [`docs/MODEL_MANIFEST.md`](docs/MODEL_MANIFEST.md) and the epyc-root wiki. Cross-repo probe rows go to the canonical scoreboard in epyc-root, never into a fork of it here — see [`docs/reference/model-probe-scoreboard.md`](docs/reference/model-probe-scoreboard.md).

---

## Running Benchmarks

```bash
# Build/refresh the question pool (writes benchmarks/prompts/question_pool.jsonl)
python3 scripts/benchmark/question_pool.py --build
python3 scripts/benchmark/question_pool.py --stats     # per-suite counts

# Quality suites. Auto-runs preflight_canonical.py at sweep start unless --skip-preflight.
python3 scripts/benchmark/run_benchmark.py --list-suites
python3 scripts/benchmark/run_benchmark.py --suite gpqa --model frontdoor --dry-run

# Host/binary/launcher preflight — run standalone after a kernel rebuild, reboot,
# or executor change. Five gates; exits 0 on PASS, 1 on FAIL.
python3 scripts/preflight_canonical.py

# THE only sanctioned llama-bench entry point. Wraps scripts/lib/canonical_recipe.py
# (single source of truth for the recipe) and holds a CPU region claim via
# epyc-orchestrator's region-lock for the whole measured window.
# --dry-run validates + prints the command without firing inference.
scripts/benchmark/bench_canonical.sh -m /path/to/model.gguf --dry-run
scripts/benchmark/bench_canonical.sh -m /path/to/model.gguf -n 512 -r 2
```

Seeding / routing evaluation lives in **epyc-orchestrator**, not here — it was moved out of this repo with the monorepo split:

```bash
# 3-way routing evaluation (frontdoor vs coder vs worker)
python3 /mnt/raid0/llm/epyc-orchestrator/scripts/benchmark/seed_specialist_routing.py \
    --3way --suites math coder general --sample-size 20 --tui
```

**Methodology guard rails** (per [wiki/benchmark-methodology.md](https://github.com/pestopoppa/epyc-root/blob/main/wiki/benchmark-methodology.md) and [agents/shared/MEASUREMENT_POLICY.md](https://github.com/pestopoppa/epyc-root/blob/main/agents/shared/MEASUREMENT_POLICY.md)):

- Benchmarks run **only** via the codified recipes (`bench_canonical.sh` / `canonical_recipe.py`) — never a hand-reconstructed `llama-bench` command. Recipe drift has silently corrupted results at least three times.
- Index results by **model + quant + flags**, never by orchestrator role (roles get reassigned; reassignment must not lose data).
- Verify speed with `llama-bench` via `bench_canonical.sh`. `run_benchmark.py` is the *quality*-suite runner.
- A number without a protocol citation is an **observation** — usable for hypotheses, never to gate a keep/revert/deploy/promote decision.
- **Deterministic replay before regeneration**: if a result can be obtained by deterministically rescoring saved inference outputs, do that instead of re-running inference; rebaseline only the axis that changed.
- Always run a sweep — never deploy without measured numbers.
- `llama-bench` defaults to `-fa 0`; **always pass `-fa 1` explicitly** for CPU decode (~8–10% swing).
- Never pipe `llama-cli` output through `grep`/`tail`/`head` — redirect to file then `cat`.
- Single-model vs NUMA-concurrent modes need **independently optimized** params; don't reuse settings across regimes.

---

## Build configuration — read this before forking or reproducing

**Our builds are tuned to one specific machine, on purpose.** This repository is a
single-host research program (EPYC 9655 + MI210, gfx90a, ROCm 6.2), and the guiding rule
for every build flag is *match what production on this host actually runs* — because a
kernel improvement that does not transfer to the production binary is not an improvement
for us. That choice trades away portability, and if you fork this work you will probably
want to trade it back.

The flags that are host-specific rather than universal:

| Flag | We use | Why, and what a fork should consider |
|---|---|---|
| `AMDGPU_TARGETS` | `gfx90a` | Our MI210. Set your own architecture. |
| `GGML_NATIVE` | `ON` | Emits code tuned to this host's CPU (`-march=native`). **Portable builds should use `OFF`** — a NATIVE=ON binary may not run, or may not be comparable, on a different CPU. We take ON because every reference build on this host is ON and matching production is what makes a measured win transferable *here*. |
| `GGML_HIP_ROCWMMA_FATTN` | `ON` | **Not merely a tuning choice — the CMake default (`OFF`) is unsafe on gfx90a.** With `-fa on`, the non-rocWMMA flash-attention path produces non-finite values at longer sequence lengths (measured 2026-08-27: all 12 pinned long prompts failed with `non-finite target features`, while a 25-character prompt passed on the same binary — prompt length is the discriminator, so short smoke tests hide it). If you build for gfx90a with flash attention, turn this ON. |
| `GGML_HIP_MMQ_MFMA` | CMake default (`ON`) | Left at the default. Note our own screens found OFF faster on one small-model prefill surface, and *inverting* on MoE workloads at low batch — treat it as workload-dependent, not a global default. |

Numbers published here are therefore valid **for this hardware and these flags**. Reproducing
them on other hardware requires re-measuring, not re-reading — see [`MEASUREMENT.md`](../../MEASUREMENT.md)
in `epyc-root` for the claim grammar that governs what a measurement is allowed to assert.

The AutoKernel GPU build contract is defined in one place:
`scripts/kernel_rnd/autokernel/controller/discovery_deployment_factory.py` (`cmake_defines`).

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
│   │   ├── kernel/                        # kernel implementation maps
│   │   ├── models/                        # QUIRKS, MODELS, admission checkpoints
│   │   ├── GENERATED_DOCS_INDEX.md        # generated
│   │   └── ANALYSIS_REPORTS_INDEX.md      # generated
│   ├── data/                              # dated evidence packets + measurement manifests
│   ├── design/                            # measurement/harness specs (written before running)
│   └── experiments/                       # per-experiment analyses
│
├── handoffs/
│   └── active/master-handoff-index.md     # outstanding research-repo work
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
│   ├── benchmark/                         # ~290 scripts: question_pool, bench_canonical.sh,
│   │                                      #   run_benchmark, *_adapter, *scorer, eval_*, score_*
│   ├── lib/canonical_recipe.py            # single source of truth for the bench recipe
│   ├── preflight_canonical.py             # 5-gate host/binary/launcher preflight
│   ├── analysis/  docs/  seal/  session/  validate/  utils/  ...
│   └── (no server/ dir — llama-server lifecycle lives in epyc-orchestrator)
│
├── orchestration/
│   ├── model_registry.yaml                # comprehensive benchmark-record registry
│   ├── optimization_checkpoint.yaml
│   ├── optimization_report.md
│   ├── orchestrator_baseline.json
│   └── optuna_study.db                    # NumericSwarm hyperparameter search DB
│
├── artifacts/                             # sealed evidence bundles + attestations
├── benchmarks/                            # prompts/, results/, evidence/, images/
├── configs/                               # benchmark config templates
└── data/                                  # per-campaign raw capture + scoring fixtures
```

---

## Cross-Repo Companions

| Repo | Local path | What it owns |
|---|---|---|
| **[epyc-orchestrator](https://github.com/pestopoppa/epyc-orchestrator)** | `/mnt/raid0/llm/epyc-orchestrator` | Production orchestration that consumes these benchmarks via the autopilot eval tower. Also owns `scripts/server/` (llama-server lifecycle via `orchestrator_stack.py`), `scripts/benchmark/seed_*.py` (routing/seeding), `scripts/region-lock`, and the **lean** registry compiled from this repo's full one |
| **[epyc-root](https://github.com/pestopoppa/epyc-root)** | `/mnt/raid0/llm/epyc-root` | Governance, hooks, handoffs, the compiled knowledge base (`wiki/`, `research/`), `MEASUREMENT.md`, and the canonical model-probe scoreboard |
| **[llama.cpp fork](https://github.com/pestopoppa/llama.cpp)** | `/mnt/raid0/llm/llama.cpp` | `production-consolidated-v8`, frozen 2026-07-25 at `67a433bf4` (`llama-server --version` → `10107`). Single serving kernel; `ik_llama.cpp` is deprecated as a serving path. **Production branches are immutable** — all kernel/benchmark experiments happen on `llama.cpp-experimental` branches and are promoted as a new version |

---

## License

MIT. Models are under their own licenses; see per-entry notes in [intake_index.yaml](https://github.com/pestopoppa/epyc-root/blob/main/research/intake_index.yaml).
