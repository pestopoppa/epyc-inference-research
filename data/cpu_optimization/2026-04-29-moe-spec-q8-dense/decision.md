# MoE-Spec Q8 frontdoor + Dense — Decision

## Q8 frontdoor: NOT DEPLOYABLE

All B values regress -23% to -58% on Qwen3.6-35B-A3B Q8_0 pp32 (n_expert=256). Mask overhead via ggml_argsort_top_k scales with n_expert; 256-expert overhead exceeds savings.

## Dense: N/A

Qwen3.6-27B Q8 is hybrid SSM-Dense (no MoE layers). MoE-Spec doesn't fire.

## Production decision

Only **REAP-246B-A35B Q4_K_M** with `LLAMA_ARG_MOE_SPEC_BUDGET=40` qualifies for production deployment. All other models keep MoE-Spec OFF.

## Structural finding

MoE-Spec scales as O(n_expert × log B) per layer. Deployment threshold:
- ≤80 experts (REAP): GO
- 128 experts (Coder): MARGINAL (noise-dominated)
- ≥256 experts (Q8): NO-GO (overhead dominates)

The MoE-Spec mechanism is fundamentally a small-n_expert + heavy-compute mechanism. Future MoE models (e.g., Qwen3-Next-80B, GLM-5.1) should be classified by n_expert before assuming they benefit.
