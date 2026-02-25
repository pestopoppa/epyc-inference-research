# Strategic Action Plan: Next Phase of EPYC Inference Optimization

**Date:** December 15, 2025  
**Based on:** Current benchmark results, Opinion A, Opinion B  
**Current State:** 12.7x max speedup (Prompt Lookup), 11x (External Draft), +21-52% (MoE Soft Mask)

---

## Executive Summary

Both opinions converge on three critical insights:

1. **You've hit the memory bandwidth ceiling** — Further algorithmic improvements must focus on *acceptance rates* and *reducing memory movement*, not raw compute
2. **SuffixDecoding is the highest-ROI unexplored track** — Expected 2-4x additional gains on agentic/structured workloads with minimal implementation effort
3. **Hugepages + NUMA refinement are low-hanging fruit** — Guaranteed 10-15% baseline improvement with system-level changes

### Synthesized Priority Stack

| Priority | Action | Expected Gain | Effort | Rationale |
|----------|--------|---------------|--------|-----------|
| **P0** | Hugepage reservation (256-512GB of 1G pages) | +10-15% baseline | 1 hour | TLB misses are a hidden cost at model scale |
| **P0** | NUMA pinning refinement | +5-10% consistency | 1 hour | Switch from `--interleave=all` to explicit binding |
| **P1** | SuffixDecoding implementation | +100-200% on agentic | 1-2 days | Zero-cost drafting for repetitive patterns |
| **P2** | Q2_K draft quantization | +20-30% draft speed | 2 hours | Lighter drafts = faster speculation rounds |
| **P3** | Hybrid Drafting (Suffix → Draft fallback) | Combines P1 + existing | 1 day | After SuffixDecoding works, chain with external draft |
| **Deprioritized** | ZenDNN | Minimal ROI | High | AOCL BLIS already optimized |
| **Deprioritized** | PARD pipeline | Complex | High | Async speculation adds latency on CPU |

---

## Track Reorganization

### Current Track Status (From Your Research)

| Track | Status | Proven Speedup | Next Action |
|-------|--------|----------------|-------------|
| Track 1 | ✅ Production | 5.9-11x | Q2_K draft testing |
| Track 2 | ✅ Production | +21-52% | Complete (3 experts optimal) |
| Track 3 | ❌ Deprecated | 0% acceptance | Do not revisit |
| Track 6 | 🆕 Untested | +100-200% expected | **IMPLEMENT NOW** |
| Track 7 | ❌ Failed | 0.446% accept | Do not revisit |
| Track 8 | ✅ Production | 8.6-12.7x | Hybrid with Track 6 |

### New Track Structure (Aligned with Opinions)

#### **Track A: Foundational Stability** (Immediate)
System-level optimizations with guaranteed ROI.

```bash
# 1. Reserve explicit hugepages (1G × 256-512 pages for ~260-530GB)
# Add to /etc/default/grub:
GRUB_CMDLINE_LINUX="hugepagesz=1G hugepages=300 default_hugepagesz=1G"
sudo update-grub && reboot

# 2. Verify hugepage allocation
cat /proc/meminfo | grep -i huge
# HugePages_Total: 300
# HugePages_Free: 300

# 3. Refined NUMA pinning (replace --interleave=all)
numactl --physcpubind=0-95 --membind=0-11 llama-speculative ...
```

**Measurement:** Baseline before/after on Qwen2.5-Coder-32B. Target: +10-15% t/s.

#### **Track B: SuffixDecoding** (High Priority)
Both opinions agree this is the highest-ROI algorithmic advancement.

**Why SuffixDecoding over continued Track 1 tuning:**
- Track 1 (external draft) is already near-optimal at 11x
- SuffixDecoding provides **zero-cost drafting** for patterns already seen
- Ideal for your target workloads: agentic (JSON schemas, function calls), code (boilerplate), SQL

**Implementation Strategy:**
1. **Phase 1:** Standalone suffix tree in llama.cpp (C++ port from reference)
2. **Phase 2:** Hybrid fallback: `Suffix lookup → External draft → Full model`
3. **Phase 3:** Global + per-request trees (shared patterns + conversation context)

**Expected Performance:**
| Workload | Current Best | SuffixDecoding | Combined Potential |
|----------|-------------|----------------|-------------------|
| Summarization | 95.2 t/s (Lookup) | ~120 t/s | ~130 t/s |
| Code generation | 33.0 t/s (K=24) | ~60 t/s | ~80 t/s |
| Agentic/JSON | Not tested | ~4.5x expected | ~150 t/s potential |

#### **Track C: Draft Model Optimization** (Medium Priority)
Squeeze more speed from proven Track 1 approach.

**Action 1: Q2_K Draft Quantization**
```bash
# Convert Qwen2.5-0.5B to Q2_K
./llama-quantize \
  /mnt/raid0/llm/models/lmstudio-community/Qwen2.5-0.5B-Instruct-Q8_0.gguf \
  /mnt/raid0/llm/models/Qwen2.5-0.5B-Q2_K.gguf \
  Q2_K

# Benchmark: Compare draft speeds
llama-bench -m Qwen2.5-0.5B-Q8_0.gguf -m Qwen2.5-0.5B-Q2_K.gguf -t 96
```

**Hypothesis:** Q2_K draft at ~100+ t/s (vs 85 t/s Q8_0) enables faster speculation rounds.

**Action 2: Qwen3-0.6B Draft Testing**
Test Qwen3-0.6B as draft for Qwen3-32B (currently 3.1x speedup with 39.1% acceptance):
```bash
# Convert to Q2_K
./llama-quantize Qwen3-0.6B-Q8_0.gguf Qwen3-0.6B-Q2_K.gguf Q2_K

# Test against Qwen3-32B
llama-speculative \
  -m Qwen3-32B-Q4_K_M.gguf \
  -md Qwen3-0.6B-Q2_K.gguf \
  --draft-max 16 -t 96
```

---

## Discarded/Deprioritized Items

### From Opinion A & B Consensus:

| Item | Status | Reason |
|------|--------|--------|
| ZenDNN | **Dropped** | AOCL BLIS already provides +5-11%; further BLAS tuning is diminishing returns |
| EAGLE (Track 3) | **Dropped** | 20+ hours spent, 0% acceptance; quantization mismatch is fundamental |
| Medusa | **Deferred** | Requires training per model; not worth effort vs SuffixDecoding |
| CAS-Spec/CLaSp | **Deferred** | Your Track 7 test showed 0.446% acceptance; needs trained exit classifiers |
| PARD Pipeline | **Deprioritized** | Async parallelization adds complexity; CPU memory latency makes pipelining less effective than on GPU |
| Prompt Engineering | **Lowest** | Task-specific; system-wide algorithmic improvements yield higher ROI |

### From Your Research:

| Track | Status | Reason |
|-------|--------|--------|
| Track 3 (EAGLE-1) | **Deprecated** | Quantization mismatch; SpecMQuant checkpoints are the only path forward (low priority) |
| Track 4 (Medusa) | **Skipped** | Training required |
| Track 5 (SSM Speculation) | **Blocked** | Fundamental architectural incompatibility |
| Track 9 (CLaSp/SWIFT) | **Skipped** | Same issues as CAS-Spec |
| Track 10 (Kangaroo) | **Skipped** | Adapter training required |

---

## Measurement Protocol

### Benchmark Targets

After implementing Track A + B, your target metrics should be:

| Workload | Current Best | Target (Post-Optimization) |
|----------|-------------|---------------------------|
| Summarization (MoE) | 95.2 t/s | **110-130 t/s** |
| Code generation (Dense 32B) | 33.0 t/s | **50-70 t/s** |
| Agentic/JSON | Not tested | **100-150 t/s** |
| General inference (72B) | 8.53 t/s | **12-15 t/s** |

### Validation Checklist

Before declaring success on each track:

- [ ] **Hugepages:** `cat /proc/meminfo | grep Huge` shows allocation
- [ ] **NUMA:** `numastat -m` shows balanced memory across nodes
- [ ] **SuffixDecoding:** Test on JSON/SQL benchmark, measure pattern hit rate
- [ ] **Q2_K Draft:** Verify no quality degradation (same output as Q8_0)
- [ ] **Hybrid:** Confirm suffix lookup attempts before draft fallback

---

## Implementation Timeline

### Week 1: Foundation + Quick Wins

| Day | Task | Deliverable |
|-----|------|-------------|
| 1 | Hugepage + NUMA setup | Before/after baseline benchmarks |
| 2 | Q2_K draft quantization | Draft speed comparison table |
| 3-5 | SuffixDecoding Phase 1 | Standalone suffix tree prototype |

### Week 2: Integration + Validation

| Day | Task | Deliverable |
|-----|------|-------------|
| 1-2 | SuffixDecoding integration into llama.cpp | Working `--suffix-decode` flag |
| 3-4 | Hybrid drafting (Suffix → External Draft) | End-to-end pipeline |
| 5 | Comprehensive benchmarking | Updated research report |

### Week 3: Documentation + Publication

| Day | Task | Deliverable |
|-----|------|-------------|
| 1-2 | Results analysis and writeup | Blog post draft |
| 3-5 | GitHub repo updates, code cleanup | Public release |

---

## Key Differences Between Opinions

| Topic | Opinion A | Opinion B | Synthesis |
|-------|-----------|-----------|-----------|
| **NUMA approach** | `--physcpubind=0-95 --membind=0-11` | Same recommendation | Adopt explicit pinning |
| **Hugepages** | 256-512GB of 1G pages | Same recommendation | 300 pages (307GB) is safe starting point |
| **SuffixDecoding** | Priority 1 | Priority 1 (Track B) | Both agree — implement first |
| **PARD Pipeline** | Priority 2 | Not mentioned | **Deprioritize** — Opinion A overestimates CPU pipelining benefit |
| **Draft quantization** | Q2_K recommended | Q2_K recommended | Implement for Qwen2.5-0.5B and Qwen3-0.6B |
| **ZenDNN** | Deprioritized | Dropped | Drop entirely |
| **Hybrid Drafting** | Deferred after Suffix | Integrated with Suffix | Implement as Phase 2 of SuffixDecoding |

---

## Risks and Mitigations

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| SuffixDecoding gains are workload-specific | Medium | Low | Test across all workload types; fall back to draft |
| Hugepages cause memory fragmentation | Low | Medium | Use `hugeadm --pool-pages-min` for dynamic allocation |
| Q2_K draft hurts acceptance rate | Medium | Low | Compare against Q8_0; use Q4_K_M as fallback |
| llama.cpp integration complexity | Medium | High | Start with standalone prototype; integrate incrementally |

---

## Immediate Next Steps

1. **Today:** Implement hugepage reservation and NUMA pinning
2. **Today:** Create Q2_K versions of Qwen2.5-0.5B and Qwen3-0.6B drafts
3. **Tomorrow:** Begin SuffixDecoding C++ prototype
4. **This week:** First working suffix tree demo on JSON generation task

---

## Appendix: Command Templates

### Hugepage Setup
```bash
# Reserve 300 × 1GB hugepages (307GB)
echo 300 | sudo tee /sys/kernel/mm/hugepages/hugepages-1048576kB/nr_hugepages

# Verify
grep Huge /proc/meminfo

# Make permanent: add to /etc/sysctl.conf
vm.nr_hugepages_mempolicy=300
```

### NUMA Pinning
```bash
# Old approach (replace this)
numactl --interleave=all llama-speculative ...

# New approach (explicit binding)
numactl --physcpubind=0-95 --membind=0-11 llama-speculative ...
```

### Q2_K Quantization
```bash
./llama-quantize \
  input.gguf \
  output-Q2_K.gguf \
  Q2_K
```

### Benchmark Template
```bash
OMP_NUM_THREADS=1 numactl --physcpubind=0-95 --membind=0-11 \
  llama-speculative \
  -m /mnt/raid0/llm/models/target.gguf \
  -md /mnt/raid0/llm/models/draft-Q2_K.gguf \
  --draft-max 24 \
  -t 96 \
  -p "Your test prompt here"
```

---

*Generated from synthesis of Opinion A, Opinion B, and current research state (2025-12-15)*
