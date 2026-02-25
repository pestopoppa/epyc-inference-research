# Model Manifest

Role-based model configuration for the orchestrator. You don't need the exact models listed here — the orchestrator supports any compatible GGUF models in each role slot.

## Server Topology

| Role | Port | Model | RAM | Tier | Speed |
|------|------|-------|-----|------|-------|
| Front Door | 8080 | Qwen3-Coder-30B-A3B (Q4_K_M) | 20 GB | HOT | 47 t/s |
| Coder (escalation) | 8081 | Qwen2.5-Coder-32B (Q4_K_M) | 20 GB | HOT | 39 t/s |
| Worker (general) | 8082 | Qwen2.5-7B (f16) | 16 GB | HOT | 44 t/s |
| Architect (general) | 8083 | Qwen3-235B-A22B (Q4_K_M) | 134 GB | WARM | 6.1 t/s |
| Architect (coding) | 8084 | Qwen3-Coder-480B-A35B (Q4_K_M) | 272 GB | WARM | 9.0 t/s |
| Ingest (long context) | 8085 | Qwen3-Next-80B-A3B (Q4_K_M) | 46 GB | WARM | 6.3 t/s |

Supporting services:

| Service | Port | Model | Purpose |
|---------|------|-------|---------|
| Voice | 9000 | faster-whisper large-v3-turbo | Speech-to-text |
| Document OCR | 9001 | LightOnOCR-2-1B (Q4_K_M) | PDF/image OCR |
| Code Search | 8088 | LateOn-Code (130M ONNX) | NextPLAID code retrieval |
| Doc Search | 8089 | answerai-colbert-small-v1 (ONNX INT8) | NextPLAID doc retrieval |
| Embeddings | 8090-8095 | BGE-large-en-v1.5 (f16) | 6-instance embedding pool |

## Memory Tiers

- **HOT** (~40 GB): Always resident. Minimum for interactive use.
- **WARM** (~430 GB): mmap-preloaded, loaded on demand. Full production stack.
- **COLD**: Disk-only, loaded manually.

Start by tier:

```bash
python3 scripts/server/orchestrator_stack.py start --hot-only   # HOT only
python3 scripts/server/orchestrator_stack.py start              # HOT + WARM
```

## Substitution Guide

Each role has specific requirements. When substituting models, match these constraints:

### Front Door (Tier A)

Routes requests, generates Python code, runs tools.

- **Needs**: Fast MoE model with good instruction following
- **Acceleration**: MoE expert reduction + speculative decoding + prompt lookup
- **Compatible substitutes**: Any Qwen3 MoE model, Mixtral-family models
- **Draft model**: Must share vocabulary with target. Qwen3-Coder family uses BOS=comma (token 11) — use jukofyork vocab-transplant drafts only

### Coder (Tier B)

Code generation, refactoring, implementation tasks.

- **Needs**: Strong code model, 32B+ parameters
- **Acceleration**: Speculative decoding (K=24) + prompt lookup
- **Compatible substitutes**: DeepSeek-Coder-V2, CodeLlama-34B, StarCoder2-33B
- **Draft model**: Same family 0.5B-1.5B quantized to Q8_0

### Worker (Tier C)

Parallel file-level tasks — summaries, exploration, simple code.

- **Needs**: Fast 7B-class model
- **Acceleration**: Speculative decoding + prompt lookup
- **Compatible substitutes**: Any 7B instruction-tuned model (Llama-3, Mistral-7B, Phi-3)
- **Draft model**: 0.5B from same family

### Architect — General (Tier B)

System architecture, complex multi-step reasoning.

- **Needs**: Largest available MoE model, full expert count for quality
- **Acceleration**: Speculative decoding only (K=16). No expert reduction.
- **Compatible substitutes**: DeepSeek-V3, Llama-3-405B (dense, requires more RAM)

### Architect — Coding (Tier B)

Hardest coding problems, architecture-level code decisions.

- **Needs**: Largest code-specialized model available
- **Acceleration**: Speculative decoding only (K=16). No expert reduction.
- **Compatible substitutes**: DeepSeek-Coder-V3, any 200B+ code model

### Ingest / Long Context (Tier B)

Document summarization, long-context synthesis.

- **Needs**: SSM or hybrid architecture for efficient long-context processing
- **Acceleration**: MoE expert reduction only. **No speculative decoding** (SSM requires consecutive positions)
- **Compatible substitutes**: Mamba-family, RWKV-family, any SSM model

## Draft Model Compatibility

Speculative decoding requires the draft model to share the target model's vocabulary:

| Target Family | Compatible Draft | Notes |
|---------------|-----------------|-------|
| Qwen2.5 | Qwen2.5-0.5B / Qwen2.5-Coder-0.5B | Standard vocab match |
| Qwen3 (non-Coder) | Qwen3-0.6B | Standard Qwen3 vocab |
| Qwen3-Coder | jukofyork-Qwen3-Coder-0.75B | Vocab-transplant draft (BOS=comma) |
| Llama-3 | Llama-3-1B | Same tokenizer family |
| Mistral | Mistral-0.5B (if available) | Same tokenizer |

Run `/draft-compat` to validate draft-target compatibility for a specific pair.

## Downloading Models

Models with `huggingface_id` in the registry can be downloaded automatically:

```bash
python scripts/setup/download_models.py --tier hot    # Minimum set
python scripts/setup/download_models.py --tier warm   # Full production
python scripts/setup/download_models.py --tier all    # Everything
```

## Configuration

All model configuration lives in `orchestration/model_registry.yaml`. Key sections:

- `runtime_defaults` — quantization, thread count, NUMA policy, context limits
- `server_mode` — per-role server definitions with ports, models, acceleration
- `roles` — detailed role definitions with launch commands and quirks

See the [registry file](../orchestration/model_registry.yaml) for the complete configuration.
