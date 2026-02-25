# Model Reference

Comprehensive reference for all models used in the orchestration system.

**Last verified:** 2026-02-03 against `orchestration/model_registry.yaml`

## Production Models by Role

### Tier A - Front Door / Orchestrator

| Role | Model | Quant | Port | Speed | Acceleration |
|------|-------|-------|------|-------|--------------|
| frontdoor | Qwen3-Coder-30B-A3B | Q4_K_M | 8080 | 18.3 t/s | MoE 6 experts |

### Tier B - Specialists

| Role | Model | Quant | Port | Speed | Acceleration |
|------|-------|-------|------|-------|--------------|
| coder_escalation | Qwen2.5-Coder-32B | Q4_K_M | 8081 | 39.4 t/s | Spec K=24 + lookup |
| architect_general | Qwen3-235B-A22B | Q4_K_M | 8083 | 6.75 t/s | MoE 4 experts |
| architect_coding | Qwen3-Coder-480B-A35B | Q4_K_M | 8084 | 10.3 t/s | MoE 3 experts |
| ingest_long_context | Qwen3-Next-80B-A3B | Q4_K_M | 8085 | 6.3 t/s | MoE 4 experts (NO SPEC!) |

### Tier C - Workers

| Role | Model | Quant | Port | Speed | Acceleration |
|------|-------|-------|------|-------|--------------|
| worker_general | Qwen2.5-7B-Instruct | f16 | 8082 | 43.9 t/s | Spec K=24 + lookup |
| worker_math | *(shared with worker_general)* | — | 8082 | 43.9 t/s | Spec K=24 + lookup |
| worker_vision | Qwen2.5-VL-7B | Q4_K_M | 8086 | ~15 t/s | None (VL model) |
| vision_escalation | Qwen3-VL-30B-A3B | Q4_K_M | 8087 | ~10 t/s | MoE 4 experts |

### Tier D - Draft / Embedder

| Role | Model | Quant | Port | Purpose |
|------|-------|-------|------|---------|
| draft | Qwen2.5-Coder-0.5B-Instruct | Q8_0 | — | Speculative decoding |
| embedder (6x) | BGE-large-en-v1.5 | F16 | 8090-8095 | MemRL episodic memory (probe-first) |

## Memory Footprint

| Tier | Components | Memory |
|------|------------|--------|
| HOT (always resident) | frontdoor, coder_escalation, worker, embedder | ~55 GB |
| WARM (load on demand) | architects, ingest, vision | ~450 GB |
| Total available | All models loaded | ~505 GB |

## Model Compatibility Matrix

### Speculative Decoding Pairs

| Target Family | Compatible Drafts | K Value | Speedup |
|---------------|-------------------|---------|---------|
| Qwen2.5-Coder-32B | Qwen2.5-Coder-0.5B-Instruct | K=24 | 5.4x (w/ lookup) |
| Qwen2.5-7B-Instruct | Qwen2.5-Coder-0.5B | K=24 | 3.8x (w/ lookup) |
| Qwen3-* (non-coder) | **None** | — | Use MoE reduction |

### Incompatible Pairs (Do Not Use)

| Target | Draft | Failure Mode |
|--------|-------|--------------|
| Qwen3-Coder-480B | Any | BOS token mismatch (`BOS=','`) |
| Qwen3-Next-* | Any | SSM state corruption |
| Qwen3-VL-* | Any text draft | Vision encoder mismatch |
| DeepSeek-R1-Distill-* | Any | Vocab size mismatch |

## MoE Override Keys

| Model Family | Override Key | Recommended Experts |
|--------------|--------------|---------------------|
| Qwen3-Coder-30B-A3B | `qwen3moe.expert_used_count` | 6 (quality), 4 (fast) |
| Qwen3-235B-A22B | `qwen3moe.expert_used_count` | 4 (balanced) |
| Qwen3-Coder-480B-A35B | `qwen3moe.expert_used_count` | 3 (memory limit) |
| Qwen3-Next (SSM) | `qwen3next.expert_used_count` | 4 (quality), 2 degrades |
| Qwen3-VL-30B-A3B | `qwen3vlmoe.expert_used_count` | 4 |

## Critical Constraints

### SSM Models (Qwen3-Next)

**NEVER use speculative decoding or prompt lookup.**

SSM architectures maintain recurrent state that cannot be rolled back. Use MoE expert reduction only.

```bash
# WRONG - will corrupt model state
llama-speculative -m Qwen3-Next-80B.gguf -md draft.gguf

# CORRECT - expert reduction only
llama-cli -m Qwen3-Next-80B.gguf --override-kv qwen3next.expert_used_count=int:4
```

### Qwen3-Coder-480B

BOS token mismatch (`BOS=','`) breaks all speculation. Use expert reduction only.

### Vision Models (VL)

Vision models require mmproj files and are incompatible with text-only draft models.

```bash
# Vision model launch
llama-qwen2vl-cli -m Qwen2.5-VL-7B-Q4_K_M.gguf \
  --mmproj mmproj-model-f16.gguf \
  --image path/to/image.png -p "Describe this image"
```

## Model Locations

```
/mnt/raid0/llm/lmstudio/models/  # Primary GGUF storage (lmstudio format)
/mnt/raid0/llm/models/           # Secondary GGUF storage
/mnt/raid0/llm/hf/               # HuggingFace format (raw)
```

## Quick Commands by Model Type

### Dense Model (with speculation + lookup)

```bash
OMP_NUM_THREADS=1 numactl --interleave=all \
  llama-speculative -m TARGET.gguf -md DRAFT.gguf \
  --draft-max 24 --lookup-ngram-min 3 -t 96 -p "prompt"
```

### MoE Model (with expert reduction)

```bash
numactl --interleave=all \
  llama-cli -m MOE_MODEL.gguf \
  --override-kv qwen3moe.expert_used_count=int:4 -t 96 -p "prompt"
```

### SSM Model (expert reduction ONLY)

```bash
numactl --interleave=all \
  llama-cli -m SSM_MODEL.gguf \
  --override-kv qwen3next.expert_used_count=int:4 -t 96 -p "prompt"
```

### Vision Model (with mmproj)

```bash
numactl --interleave=all \
  llama-qwen2vl-cli -m VL_MODEL.gguf --mmproj MMPROJ.gguf \
  --image input.png -p "Describe this image" -t 96
```

---

*See [QUIRKS.md](QUIRKS.md) for runtime issues and workarounds.*
*See [RESULTS.md](../benchmarks/RESULTS.md) for benchmark data.*
*Source of truth: `orchestration/model_registry.yaml`*
