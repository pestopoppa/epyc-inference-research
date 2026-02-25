# CPU Optimization Research Findings

**Date**: 2026-01-05
**Status**: Preliminary Investigation Complete

---

## 1. T-MAC Evaluation

### Repository
- **Location**: `/mnt/raid0/llm/T-MAC/`
- **Paper**: [arXiv:2407.00088](https://arxiv.org/abs/2407.00088)
- **Conference**: EuroSys 2025

### How T-MAC Works
T-MAC replaces dequantization with lookup tables for low-bit (1-4 bit) inference:
1. Groups one-bit weights (e.g., into groups of 4)
2. Precomputes all possible partial sums
3. Uses LUT with tbl/pshuf instructions for fast lookup
4. Reduces table size from 2^n to 2^(n-1) with sign bit

### Compatibility Assessment

| Aspect | Status | Notes |
|--------|--------|-------|
| Quantization support | Partial | W4A16 from GPTQ/gguf, but NOT Q4_K_M |
| llama.cpp version | Old | Based on b2794 (May 2024), heavily modified |
| Model conversion | Required | Needs HF → T-MAC GGUF conversion |
| x86/AVX-512 support | Uncertain | Uses TVM codegen, no explicit Zen 5 optimizations |
| Existing GGUF models | NOT compatible | Must reconvert from HuggingFace |

### Critical Warning from README
> "We have noticed many users attempting to evaluate T-MAC on old-gen x86 platforms... we cannot guarantee significant speedup (especially for 4-bit token generation) on all x86 platforms."

### Recommendation
**MEDIUM PRIORITY** - T-MAC requires:
1. Model reconversion (time-consuming)
2. Building separate llama.cpp fork with `-DLLAMA_TMAC=ON`
3. May not yield significant gains on x86 at 4-bit
4. Best gains are at 1-2 bit, which degrades quality

**Test approach**: Start with a small 2-bit GPTQ model (NOT our production models) to validate x86 performance before committing to full conversion pipeline.

---

## 2. Tree Speculation Status

### llama.cpp Support Confirmed
From `examples/speculative/speculative.cpp`:
```cpp
// max number of parallel drafting sequences (i.e. tree branches)
// sample n_draft tokens from the draft model using tree-based sampling
```

### Available Flags
```
--draft-max N       Max tokens to draft
-td, --threads-draft N
-Cd, --cpu-mask-draft M
-Crd, --cpu-range-draft lo-hi
```

### Note
Tree-based sampling is already integrated into llama.cpp's speculative binary. The `--draft-max` flag with higher values enables tree exploration. No separate `--draft-n-tree` flag found in current build.

### Recommendation
**HIGH PRIORITY** - Already available, just needs benchmarking:
1. Test `--draft-max 32` vs current `--draft-max 24`
2. Measure acceptance rate changes
3. Profile memory bandwidth at different tree widths

---

## 3. NUMA Topology (Actual)

### System Configuration
```
available: 2 nodes (0-1)
node 0 cpus: 0-47, 96-143 (48 cores + SMT)
node 0 size: 580105 MB
node 0 free: 5398 MB

node 1 cpus: 48-95, 144-191 (48 cores + SMT)
node 1 size: 580512 MB
node 1 free: 6081 MB

node distances:
       0    1
  0:  10   12
  1:  12   10
```

### Key Finding
**Only 2 NUMA nodes** (NPS1 configuration), NOT 8 as originally planned.

**Impact on Multi-Draft Parallel**:
- Original plan assumed 8 NUMA domains (NPS4 mode)
- With only 2 nodes, cannot run 3+ draft models on separate NUMA domains
- Would need BIOS reconfiguration to enable NPS4

### BIOS Option (If Needed)
NPS (NUMA Per Socket) can be reconfigured:
- NPS1: 1 NUMA domain per socket (current)
- NPS2: 2 NUMA domains per socket
- NPS4: 4 NUMA domains per socket

With NPS4 on a 2-socket system = 8 NUMA domains.

---

## 4. Multi-Draft and Orchestration Compatibility

### Question
> "Won't NUMA pinning for parallel drafting interfere with how we're thinking the orchestration engine will operate?"

### Answer: No Conflict
The orchestrator already routes to different binaries based on task type:
- `llama-speculative` for code gen
- `llama-cli` for chat
- MoE-reduced processes for reasoning

Multi-draft parallel would be **another routing target**, not a replacement:
```
Code gen task → Multi-draft speculative process (NUMA-pinned)
Chat/summarization → Standard model process
MoE reasoning → Expert-reduced process
```

### Switching Overhead
Switching between modes is at the **process level** (orchestrator already does this). No significant overhead beyond model loading (which is amortized across many requests).

---

## 5. Updated Priority Matrix

| Track | Original Priority | Revised Priority | Reason |
|-------|-------------------|------------------|--------|
| **A: T-MAC** | HIGH | MEDIUM | x86 gains uncertain, requires model reconversion |
| **B: Tree Spec Tuning** | MEDIUM | HIGH | Already available, just needs benchmarking |
| **C: Multi-Draft Parallel** | MEDIUM | LOW | Only 2 NUMA nodes (need BIOS change for NPS4) |
| **D: AVX-512 Kernels** | LOW | LOW | High effort, use devc for autonomous development |

---

## 6. Devcontainer YOLO Mode for Kernel Development

### Concept
The heavy kernel development work (Track D: AVX-512) is well-suited to run in devcontainer environment with autonomous execution:
- Agent works non-stop until success or demonstrated infeasibility
- All progress/failures documented for external review
- No interactive prompts needed

### Implementation
1. Use existing devc setup at `/mnt/raid0/llm/tools/devc/`
2. Create a separate workspace for kernel experiments
3. Agent logs to progress file every N iterations
4. External review via `tail -f /mnt/raid0/llm/claude/research/kernel_dev_progress.log`

### Success Criteria
- Measurable tok/s improvement on synthetic benchmark
- No quality regression (perplexity unchanged)
- Code compiles and runs without segfaults

### Failure Criteria
- No measurable improvement after X iterations
- Hardware limitation identified (e.g., memory bandwidth ceiling)
- Breaking change to ggml that can't be upstreamed

---

## 7. Immediate Next Steps

### Safe to Execute Now (No Model Interference)

1. **Tree speculation benchmarking** (HIGH PRIORITY)
   ```bash
   # Wait for current benchmark to complete, then:
   llama-speculative -m Qwen2.5-Coder-32B.gguf -md 0.5B.gguf \
       --draft-max 32 -t 96 -p "test prompt"
   # Compare acceptance rate and throughput to --draft-max 24
   ```

2. **T-MAC 2-bit test** (MEDIUM PRIORITY)
   ```bash
   # Download a small 2-bit GPTQ model for testing
   # Test T-MAC performance on x86 before committing to full pipeline
   ```

3. **Profile memory bandwidth ceiling**
   ```bash
   # Run STREAM benchmark to establish baseline
   # Compare to actual inference memory bandwidth
   ```

### Requires BIOS/Hardware Change

- NPS4 configuration for multi-draft parallel (deferred)

### Long-Term (Devc YOLO)

- AVX-512 kernel development in isolated environment
- Automated progress logging for external review

---

## References

- T-MAC: https://github.com/microsoft/T-MAC
- T-MAC Paper: https://arxiv.org/abs/2407.00088
- llama.cpp speculative: `/mnt/raid0/llm/llama.cpp/examples/speculative/`
- R&D Plan: `/home/daniele/.claude/plans/twinkly-sniffing-crescent.md`
