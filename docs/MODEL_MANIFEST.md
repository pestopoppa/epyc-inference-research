# Model Manifest

Role-based model reference for the current orchestration stack. This research
repo records benchmark evidence and candidate history; the live deployment
truth is compiled in `epyc-orchestrator` from its lean registry, descriptors,
launch manifest, and generated stack priors.

**Live snapshot source**:
`/mnt/raid0/llm/epyc-orchestrator/orchestration/derived/stack_priors.yaml`
compiled at `2026-06-27T13:42:28Z` with `stack_priors_version: 4`.

## Live Server Topology

| Role | Endpoint | Model | Model RAM | Tier | Prior TPS | Context |
|------|----------|-------|-----------|------|-----------|---------|
| `frontdoor` | `8070` primary; launch ports `8070/8080/8180/8280/8380` | Qwen3.6-35B-A3B-Q8_0 | 37 GB | HOT | 24.3 | 32K effective / 262K max |
| `coder_escalation` | `8070` | Qwen3.6-35B-A3B-Q8_0, shared with `frontdoor` | 37 GB shared | HOT | 24.3 | 32K effective / 262K max |
| `worker_general` | `8072` primary; launch ports `8072/8082/8182/8282/8382` | gemma-4-26B-A4B-it-Q4_K_M | 16 GB | HOT | 60.7 | 16K |
| `worker_math` | `8072/8082` | shared with `worker_general` | 16 GB shared | HOT | 60.7 | 16K |
| `worker_summarize` | `8070` | shared with `frontdoor` | 37 GB shared | HOT | 24.3 | 32K effective / 262K max |
| `toolrunner` | `8072/8082` | shared with `worker_general` | 16 GB shared | HOT | 60.7 | 16K |
| `worker_vision` | `8086` | Qwen2.5-VL-7B-Instruct | 4.4 GB | HOT | 20.0 | 8K effective / 128K max |
| `vision_escalation` | `8087` primary; launch ports `8087/8187/8287/8387/8487` | Qwen3-VL-30B-A3B-Instruct | 18 GB | HOT | 27.6 | 16K effective / 262K max |
| `architect_general` | `8083` | Qwen3.5-122B-A10B | 69 GB | HOT | 12.19 | 16K effective / 262K max |
| `ingest_long_context` | `8085` primary; launch ports `8085/8185/8285/8385/8485` | Qwen3-Next-80B-A3B-Instruct | 45 GB | HOT | 20.8 | 32K effective / 262K max |

`architect_coding` is retired as a distinct live role. Legacy serialized labels
may normalize to `architect_general` in orchestrator compatibility paths, but
new routing, scoring, launch, and benchmark interpretation must not treat it as
a live server.

## Memory And Cost Semantics

- All live roles in the generated stack-prior snapshot are HOT.
- Shared mmap roles must not be double-counted by role. For example,
  `frontdoor`, `coder_escalation`, and `worker_summarize` share the same
  Qwen3.6 server family; `worker_general`, `worker_math`, and `toolrunner`
  share the Gemma worker runtime.
- `priors.memory_cost` is `1.0` for every live role in the current generated
  stack-prior contract. Consumers that need physical RAM should use the model
  descriptor `mem_gb` plus shared-server binding, not handwritten role totals.
- There is no current live WARM role in stack priors. Benchmark candidates and
  historical large models remain evidence/candidate records until explicitly
  promoted into the live orchestrator stack.

Supporting services:

| Service | Port | Model | Purpose |
|---------|------|-------|---------|
| Voice | 9000 | faster-whisper large-v3-turbo | Speech-to-text |
| Document OCR | 9001 | LightOnOCR-2-1B (Q4_K_M) | PDF/image OCR |
| Code Search | 8088 | LateOn-Code (130M ONNX) | NextPLAID code retrieval |
| Doc Search | 8089 | answerai-colbert-small-v1 (ONNX INT8) | NextPLAID doc retrieval |
| Embeddings | 8090-8095 | BGE-large-en-v1.5 (f16) | Embedding pool |

## Current Model Candidates And Evidence

- **Qwen3-Coder-REAP-246B-A35B (Q4_K_M)**: 50%-pruned architect candidate,
  about 139 GB, measured around 6.25 t/s in earlier research probes.
- **DeepSeek-V3 and other large architects**: candidate/evidence records only
  unless promoted through the stack-change pipeline.
- **Historical Qwen2.5/Qwen3-Coder rows**: retained for benchmark comparison,
  draft-compatibility notes, and regression context; they are not live-role
  truth unless present in generated stack priors.

## Substitution Guide

Each role has specific requirements. When substituting models, update structured
orchestrator truth first, then compile descriptors and stack priors.

### Front Door / Coder Escalation

Routes requests, writes code, and handles escalation paths through the shared
Qwen3.6 server.

- **Needs**: Fast MoE model with strong instruction following and code ability.
- **Acceleration**: Current live stack records no speculative decoding for this
  server; use generated launch requirements rather than copying old draft
  settings.
- **Compatibility risk**: Frontdoor and coder escalation share a physical model
  and endpoint, so cost, memory, and health accounting must be alias-aware.

### Worker Roles

Parallel file-level tasks, summaries, exploration, tool calls, and simple code
paths use the Gemma worker runtime or a shared frontdoor summarization path.

- **Needs**: Fast instruction-following model with reliable tool compliance.
- **Acceleration**: Current Gemma worker records MTP acceleration in generated
  stack priors.
- **Compatibility risk**: `worker_math` and `toolrunner` are shared-runtime
  aliases, not separate live model processes.

### Architect General

System architecture, deep multi-step reasoning, and high-stakes planning.

- **Needs**: Largest available high-quality reasoning MoE model.
- **Acceleration**: Current stack uses MoE expert reduction; generated runtime
  witness disables speculative decoding for this role.
- **Compatibility risk**: Do not reintroduce a distinct `architect_coding`
  server without a full stack-change update and guard pass.

### Ingest / Long Context

Long-context document synthesis and ingestion.

- **Needs**: SSM or hybrid architecture for efficient long-context processing.
- **Acceleration**: MoE expert reduction only. SSM state makes speculative
  decoding unsafe unless future evidence and launcher support explicitly prove
  otherwise.

## Draft Model Compatibility

Speculative decoding requires the draft model to share the target model's
vocabulary and runtime assumptions. Treat this table as compatibility guidance,
not a statement of what is currently enabled in production.

| Target Family | Compatible Draft | Notes |
|---------------|------------------|-------|
| Qwen2.5 | Qwen2.5-0.5B / Qwen2.5-Coder-0.5B | Standard vocab match |
| Qwen3 non-Coder | Qwen3-0.6B | Standard Qwen3 vocab |
| Qwen3-Coder | jukofyork-Qwen3-Coder-0.75B | Vocab-transplant draft; BOS=comma |
| Gemma worker | Generated MTP requirements | Use stack-prior launch requirements |
| SSM / Qwen3-Next | None by default | Speculation can corrupt recurrent state |
| Vision models | None by default | Vision encoder/mmproj path must match |

Run `/draft-compat` to validate draft-target compatibility for a specific pair.

## Configuration

Live deployment configuration is validated in `epyc-orchestrator`:

- `orchestration/model_registry.yaml` for lean live topology.
- `orchestration/model_descriptors.yaml` for model identity and measured
  evidence.
- `orchestration/derived/stack_priors.yaml` for generated consumer contracts.
- `scripts/registry/stack_change_pipeline.py check --run-promotion-gate` before
  launch, AutoPilot promotion, or benchmark interpretation.

This research repo's `orchestration/model_registry.yaml` remains useful for
comprehensive benchmark history and candidate evidence. Do not treat it as the
live launch source without reconciling through the orchestrator stack-change
pipeline.
