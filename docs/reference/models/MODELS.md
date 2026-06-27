# Model Reference

Comprehensive model reference for the orchestration system.

**Last live-stack verification:** 2026-06-27 against
`/mnt/raid0/llm/epyc-orchestrator/orchestration/derived/stack_priors.yaml`
compiled at `2026-06-27T18:56:04Z`.

This document is a research-facing snapshot. The current live stack is governed
by orchestrator generated stack priors; historical benchmark rows in this repo
are evidence, not launch truth.

## Production Models By Role

### Front Door And Coder Escalation

| Role | Model | Quant | Endpoint / Ports | Prior TPS | Acceleration |
|------|-------|-------|------------------|-----------|--------------|
| `frontdoor` | Qwen3.6-35B-A3B-Q8_0 | Q8_0 | `8070` primary; `8070/8080/8180/8280/8380` launch ports | 24.3 | none |
| `coder_escalation` | Qwen3.6-35B-A3B-Q8_0, shared with `frontdoor` | Q8_0 | `8070` | 24.3 | none |

### Workers And Tools

| Role | Model | Quant | Endpoint / Ports | Prior TPS | Acceleration |
|------|-------|-------|------------------|-----------|--------------|
| `worker_general` | gemma-4-26B-A4B-it-Q4_K_M | Q4_K_M | `8072` primary; `8072/8082/8182/8282/8382` launch ports | 60.7 | MTP |
| `worker_math` | shared with `worker_general` | Q4_K_M | `8072/8082` | 60.7 | MTP |
| `toolrunner` | shared with `worker_general` | Q4_K_M | `8072/8082` | 60.7 | MTP |
| `worker_summarize` | shared with `frontdoor` | Q8_0 | `8070` | 24.3 | none |

### Vision

| Role | Model | Quant | Endpoint / Ports | Prior TPS | Acceleration |
|------|-------|-------|------------------|-----------|--------------|
| `worker_vision` | Qwen2.5-VL-7B-Instruct | Q4_K_M | `8086` | 20.0 | baseline VL |
| `vision_escalation` | Qwen3-VL-30B-A3B-Instruct | Q4_K_M | `8087` primary; `8087/8187/8287/8387/8487` launch ports | 27.6 | MoE expert reduction |

### Architect And Ingest

| Role | Model | Quant | Endpoint / Ports | Prior TPS | Acceleration |
|------|-------|-------|------------------|-----------|--------------|
| `architect_general` | Qwen3.5-122B-A10B | Q4_K_M | `8083` | 12.19 | MoE expert reduction |
| `ingest_long_context` | Qwen3-Next-80B-A3B-Instruct | Q4_K_M | `8085` primary; `8085/8185/8285/8385/8485` launch ports | 20.8 | MoE expert reduction |

### Auxiliary Services

| Role | Model | Port | Purpose |
|------|-------|------|---------|
| `draft` | role-specific generated requirements | - | Speculative/MTP support where stack priors enable it |
| `embedder` | BGE-large-en-v1.5 | 8090-8095 | Embedding pool |
| `code_search` | LateOn-Code ONNX | 8088 | NextPLAID code retrieval |
| `doc_search` | answerai-colbert-small-v1 ONNX INT8 | 8089 | NextPLAID doc retrieval |

## Role Lifecycle Notes

- `architect_coding` is retired as a distinct live role. Legacy labels may map
  to `architect_general` in compatibility paths, but live docs, routing,
  scoring, and benchmarks should not advertise a separate server.
- `frontdoor`, `coder_escalation`, and `worker_summarize` share a physical
  Qwen3.6 runtime.
- `worker_general`, `worker_math`, and `toolrunner` share the Gemma worker
  runtime.
- All roles above are HOT in the current generated stack-prior snapshot. There
  is no active WARM live role in that contract.

## Memory Footprint

Use generated serving bindings when calculating memory pressure. The descriptor
`mem_gb` field describes the physical model; `priors.memory_cost` describes the
current routing/scoring cost. Shared mmap roles must not be added together as if
each role owned a separate model.

| Physical Runtime | Roles | Descriptor Memory |
|------------------|-------|-------------------|
| Qwen3.6-35B-A3B-Q8_0 | `frontdoor`, `coder_escalation`, `worker_summarize` | 37 GB |
| gemma-4-26B-A4B-it-Q4_K_M | `worker_general`, `worker_math`, `toolrunner` | 16 GB |
| Qwen2.5-VL-7B-Instruct | `worker_vision` | 4.4 GB |
| Qwen3-VL-30B-A3B-Instruct | `vision_escalation` | 18 GB |
| Qwen3.5-122B-A10B | `architect_general` | 69 GB |
| Qwen3-Next-80B-A3B-Instruct | `ingest_long_context` | 45 GB |

## Model Compatibility Matrix

### Speculative / Draft Compatibility

| Target Family | Compatible Drafts | Notes |
|---------------|-------------------|-------|
| Gemma worker | generated MTP requirements | Use orchestrator stack-prior launch requirements |
| Qwen2.5 | Qwen2.5-0.5B / Qwen2.5-Coder-0.5B | Historical compatibility row |
| Qwen3 non-Coder | Qwen3-0.6B | Historical compatibility row |
| Qwen3-Coder | jukofyork-Qwen3-Coder-0.75B | Vocab-transplant draft; BOS=comma |

### Incompatible Pairs

| Target | Draft | Failure Mode |
|--------|-------|--------------|
| Qwen3-Next / SSM | Any by default | Recurrent state cannot be rolled back safely |
| Vision models | Text-only drafts | Vision encoder/mmproj mismatch |
| Qwen3-Coder family | General drafts | BOS/tokenizer mismatch |
| DeepSeek-R1-Distill family | Unmatched drafts | Vocab size mismatch |

## MoE Override Keys

Treat this as family-level runtime guidance. Current live launch values are
recorded in orchestrator stack-prior `serving.launch.runtime.flags.override_kv`.

| Model Family | Override Key | Typical Expert Setting |
|--------------|--------------|------------------------|
| Qwen3.5 MoE | `qwen35moe.expert_used_count` | 8 for current architect |
| Qwen3-Next | `qwen3next.expert_used_count` | 4 for quality-oriented ingest |
| Qwen3-VL MoE | `qwen3vlmoe.expert_used_count` | 4 for vision escalation |
| Qwen3-Coder MoE | `qwen3moe.expert_used_count` | Historical/candidate only unless promoted |

## Critical Constraints

### SSM Models

Do not use speculative decoding or prompt lookup for Qwen3-Next-style SSM models
unless a future stack-change record includes explicit runtime evidence and
launcher support. The recurrent state cannot be safely rolled back under normal
draft rejection semantics.

```bash
# Wrong for SSM state.
llama-speculative -m Qwen3-Next-80B.gguf -md draft.gguf

# Use expert reduction/runtime flags from generated stack priors.
llama-cli -m Qwen3-Next-80B.gguf --override-kv qwen3next.expert_used_count=int:4
```

### Vision Models

Vision models require matching mmproj/projector assets and are incompatible with
text-only draft models.

```bash
llama-qwen2vl-cli -m Qwen2.5-VL-7B-Q4_K_M.gguf \
  --mmproj mmproj-model-f16.gguf \
  --image path/to/image.png -p "Describe this image"
```

## Model Locations

```text
/mnt/raid0/llm/lmstudio/models/  # Primary GGUF storage
/mnt/raid0/llm/models/           # Secondary GGUF storage
/mnt/raid0/llm/hf/               # HuggingFace format
```

## Quick Commands By Model Type

Prefer generated orchestrator launch commands for production. These examples are
research probes only.

### Dense / Draft-Enabled Probe

```bash
OMP_NUM_THREADS=1 numactl --interleave=all \
  llama-speculative -m TARGET.gguf -md DRAFT.gguf \
  --draft-max 24 --lookup-ngram-min 3 -t 96 -p "prompt"
```

### MoE Probe

```bash
numactl --interleave=all \
  llama-cli -m MOE_MODEL.gguf \
  --override-kv qwen3moe.expert_used_count=int:4 -t 96 -p "prompt"
```

### SSM Probe

```bash
numactl --interleave=all \
  llama-cli -m SSM_MODEL.gguf \
  --override-kv qwen3next.expert_used_count=int:4 -t 96 -p "prompt"
```

### Vision Probe

```bash
numactl --interleave=all \
  llama-qwen2vl-cli -m VL_MODEL.gguf --mmproj MMPROJ.gguf \
  --image input.png -p "Describe this image" -t 96
```

---

*See [QUIRKS.md](QUIRKS.md) for runtime issues and workarounds.*
*See [RESULTS.md](../benchmarks/RESULTS.md) for benchmark data.*
*Live source of truth: `epyc-orchestrator/orchestration/derived/stack_priors.yaml`.*
