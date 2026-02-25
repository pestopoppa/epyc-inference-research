# CAS-Spec Implementation Plan for llama.cpp

## Overview

CAS-Spec (Cached Attention Self-Speculative) is a technique for self-speculative decoding where:
- **Draft phase**: Uses early layer exit (fast, lower quality)
- **Verify phase**: Uses full layers (slow, high quality)

This allows a single model to speculate against itself, eliminating the need for a separate draft model.

---

## Architecture Analysis

### Key Files
```
llama.cpp/
├── src/
│   ├── llama-cparams.h      # Context parameters - ADD n_layer_exit here
│   ├── llama-graph.h        # Graph context - n_layer used in layer loop
│   ├── llama-graph.cpp      # Graph context init - read n_layer_exit from cparams
│   └── models/              # Per-model build functions
│       ├── llama.cpp        # for (int il = 0; il < n_layer; ++il)
│       ├── qwen3vl-moe.cpp  # Same pattern
│       └── ...
├── common/
│   ├── arg.cpp              # CLI argument parsing - ADD --n-layer-exit-draft
│   └── speculative.cpp      # Speculative decoding framework
└── tools/
    └── speculative/main.cpp # llama-speculative binary
```

### Current Layer Loop Pattern

In `src/models/llama.cpp` (and all model files):

```cpp
// Line 23-138
for (int il = 0; il < n_layer; ++il) {
    // Attention + FFN for each layer
    ggml_tensor * inpSA = inpL;
    cur = build_norm(inpL, model.layers[il].attn_norm, ...);
    // ... attention computation ...
    cur = build_ffn(...);
    // ...
    inpL = cur;
}

// After loop - output norm + lm_head
cur = build_norm(cur, model.output_norm, ...);
cur = build_lora_mm(model.output, cur);
```

### llm_graph_context Initialization

In `src/llama-graph.cpp` (line 563):

```cpp
llm_graph_context::llm_graph_context(const llm_graph_params & params) :
    // ...
    n_layer          (hparams.n_layer),  // <-- This is where we override
    // ...
```

---

## Implementation Plan

### Step 1: Add n_layer_exit to Context Parameters

**File: `src/llama-cparams.h`**

```cpp
struct llama_cparams {
    uint32_t n_ctx;
    uint32_t n_ctx_seq;
    uint32_t n_batch;
    uint32_t n_ubatch;
    uint32_t n_seq_max;
    int32_t  n_threads;
    int32_t  n_threads_batch;

    // ADD THIS:
    uint32_t n_layer_exit;    // 0 = all layers, N = exit after N layers

    // ... rest of struct ...
};
```

### Step 2: Modify Graph Context Initialization

**File: `src/llama-graph.cpp`**

```cpp
llm_graph_context::llm_graph_context(const llm_graph_params & params) :
    arch             (params.arch),
    hparams          (params.hparams),
    cparams          (params.cparams),
    ubatch           (params.ubatch),
    n_embd           (hparams.n_embd),
    // MODIFY THIS LINE:
    n_layer          (cparams.n_layer_exit > 0 ? cparams.n_layer_exit : hparams.n_layer),
    n_rot            (hparams.n_rot),
    // ... rest of init ...
```

This single change will affect ALL model implementations because they all use `n_layer` from the graph context.

### Step 3: Add CLI Argument

**File: `common/arg.cpp`**

```cpp
// Add in the argument definitions section:
{
    {"--n-layer-exit"}, "N",
    string_format("exit after N layers (default: %d, 0 = compute all layers)\n"
                  "for layer skip / early exit speculation (CAS-Spec, CLaSp)",
                  params.n_layer_exit),
    [](common_params & params, const std::string & value) {
        params.n_layer_exit = std::stoi(value);
    }
}

// Also add for draft model:
{
    {"--n-layer-exit-draft"}, "N",
    string_format("exit after N layers for draft model (default: %d, 0 = compute all layers)",
                  params.speculative.n_layer_exit),
    [](common_params & params, const std::string & value) {
        params.speculative.n_layer_exit = std::stoi(value);
    }
}
```

### Step 4: Self-Speculative Init Function

**File: `common/speculative.cpp`**

Add new function for self-speculative decoding:

```cpp
// Create draft context using same model but with early exit
struct common_speculative * common_speculative_init_self(
        struct llama_context * ctx_tgt,
        int n_layer_draft) {

    // Get the model from target context
    const struct llama_model * model = llama_get_model(ctx_tgt);

    // Create new context params for draft
    struct llama_context_params ctx_params = llama_context_default_params();
    // Copy most params from target context
    ctx_params.n_ctx    = llama_n_ctx(ctx_tgt);
    ctx_params.n_batch  = llama_n_batch(ctx_tgt);
    // Set early exit
    ctx_params.n_layer_exit = n_layer_draft;

    // Create draft context with same model
    struct llama_context * ctx_dft = llama_new_context_with_model(model, ctx_params);

    if (!ctx_dft) {
        LOG_ERR("failed to create draft context for self-speculative decoding\n");
        return nullptr;
    }

    // Use existing speculative init
    return common_speculative_init(ctx_tgt, ctx_dft);
}
```

### Step 5: Add Self-Speculative CLI Flag

**File: `tools/speculative/main.cpp`**

```cpp
// In argument parsing:
if (params.n_layer_exit_draft > 0) {
    // Self-speculative mode
    spec = common_speculative_init_self(ctx, params.n_layer_exit_draft);
} else if (!params.model_draft.empty()) {
    // External draft model mode
    ctx_dft = llama_init_from_model(model_dft, ctx_params_dft);
    spec = common_speculative_init(ctx, ctx_dft);
}
```

---

## Testing Plan

### Phase 1: Basic Layer Exit
```bash
# Test layer exit alone (will produce garbage - expected)
llama-cli -m model.gguf --n-layer-exit 32 -p "test"
```

### Phase 2: Self-Speculative
```bash
# Test self-speculative with N layers for draft
llama-speculative -m model.gguf --n-layer-exit-draft 32 -p "test"
```

### Phase 3: Benchmarking
```bash
# Compare speeds at different draft layer counts
for N in 16 24 32 48; do
    llama-speculative -m Qwen2.5-72B.gguf --n-layer-exit-draft $N -n 200 -p "prompt"
done
```

---

## Expected Results

Based on CAS-Spec paper:
- **Speedup**: 1.8-2.3x on dense models
- **Optimal draft layers**: ~50% of full model
- **Quality**: Preserved (verification by full model)

### Why It Works
1. Draft phase (N layers) produces candidates quickly
2. Full model verifies each candidate
3. Accepted tokens are identical to full model output
4. Rejected tokens are regenerated with full model

---

## Risks and Mitigations

### Risk 1: KV Cache Conflicts
Both draft and verify use same model → shared KV cache
- **Mitigation**: Use separate sequence IDs for draft vs verify

### Risk 2: Memory Overhead
Two contexts for same model → ~2x context memory
- **Mitigation**: Share model weights, only duplicate KV cache

### Risk 3: Layer Output Mismatch
Early exit produces different hidden states
- **Mitigation**: Output norm is applied after layer loop, ensuring consistent format

---

## Files to Modify (Summary)

| File | Change |
|------|--------|
| `src/llama-cparams.h` | Add `n_layer_exit` field |
| `src/llama-cparams.cpp` | Initialize `n_layer_exit = 0` |
| `src/llama-graph.cpp` | Use `n_layer_exit` in `llm_graph_context` init |
| `common/arg.cpp` | Add `--n-layer-exit` and `--n-layer-exit-draft` |
| `common/speculative.cpp` | Add `common_speculative_init_self()` |
| `tools/speculative/main.cpp` | Handle self-speculative mode |

---

## Timeline Estimate

- **Day 1**: Core layer exit (Steps 1-3)
- **Day 2**: Speculative integration (Steps 4-5)
- **Day 3**: Testing and tuning

---

## Alternative: Quick Test Without Code Changes

The `--n-layer-exit` flag already exists! We can test the concept by:

1. Creating two contexts manually
2. Running draft with `--n-layer-exit N`
3. Running verify without the flag

This won't give full speedup (no KV reuse) but validates the approach.

---

## References

- CAS-Spec Paper: https://arxiv.org/abs/2510.26843
- CLaSp Paper: https://arxiv.org/abs/2505.24196
- llama.cpp speculative: `common/speculative.cpp`
- llama.cpp model builds: `src/models/*.cpp`
