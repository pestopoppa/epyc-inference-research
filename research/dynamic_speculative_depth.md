# Experiment: Dynamic Speculative Depth Control

## Hypothesis
Fixed speculative depth (K) is suboptimal. Adapting K based on recent acceptance rate and token context can improve throughput by 10-30%.

## Background
Current approach: `--speculative 8` (fixed)
Problem: Optimal K varies by:
- Draft model accuracy (varies by context)
- Token type (code vs prose vs math)
- Position in generation (early vs late)

## Implementation Approach

### Phase 1: Acceptance Rate Tracking Wrapper

Create a wrapper around llama.cpp that:
1. Runs inference with llama-server (not llama-cli)
2. Monitors acceptance rate via server metrics
3. Dynamically adjusts `--draft-max` via API or restarts

**File:** `/mnt/raid0/llm/UTILS/adaptive_speculative.py`

```python
#!/usr/bin/env python3
"""
Adaptive Speculative Decoding Controller
Adjusts speculation depth based on rolling acceptance rate
"""

import subprocess
import time
import re
from collections import deque
from dataclasses import dataclass
from typing import Optional

@dataclass
class SpecConfig:
    k_min: int = 4
    k_max: int = 24
    k_current: int = 8
    target_acceptance: float = 0.55  # Sweet spot
    adjustment_interval: int = 64    # Tokens between adjustments
    acceptance_window: int = 128     # Rolling window size

class AdaptiveSpecController:
    def __init__(self, config: SpecConfig = None):
        self.config = config or SpecConfig()
        self.acceptance_history = deque(maxlen=self.config.acceptance_window)
        self.tokens_since_adjustment = 0
    
    def record_acceptance(self, accepted: int, drafted: int):
        """Record acceptance from a speculation batch"""
        if drafted > 0:
            rate = accepted / drafted
            self.acceptance_history.append(rate)
            self.tokens_since_adjustment += drafted
    
    def get_rolling_acceptance(self) -> float:
        if not self.acceptance_history:
            return 0.5
        return sum(self.acceptance_history) / len(self.acceptance_history)
    
    def should_adjust(self) -> bool:
        return self.tokens_since_adjustment >= self.config.adjustment_interval
    
    def compute_new_k(self) -> int:
        """Adjust K based on acceptance rate"""
        rate = self.get_rolling_acceptance()
        k = self.config.k_current
        
        if rate > 0.65:
            # High acceptance - be more aggressive
            k = min(k + 4, self.config.k_max)
        elif rate > 0.55:
            # Good acceptance - slight increase
            k = min(k + 2, self.config.k_max)
        elif rate < 0.35:
            # Poor acceptance - back off significantly
            k = max(k - 4, self.config.k_min)
        elif rate < 0.45:
            # Below target - decrease
            k = max(k - 2, self.config.k_min)
        
        self.config.k_current = k
        self.tokens_since_adjustment = 0
        return k
    
    def get_k_for_context(self, prompt_snippet: str) -> int:
        """Heuristic adjustment based on token type"""
        base_k = self.config.k_current
        
        # Code detection
        code_chars = sum(1 for c in prompt_snippet if c in '{}()[];:')
        if code_chars > len(prompt_snippet) * 0.05 or '```' in prompt_snippet:
            # Code context - drafts often accurate, allow larger K
            return min(base_k + 8, self.config.k_max)
        
        # Math/formal detection
        math_indicators = ['∑', '∀', '∃', '→', '⇒', 'therefore', 'proof', '∫']
        if any(ind in prompt_snippet for ind in math_indicators):
            # Math context - drafts less reliable
            return max(base_k - 4, self.config.k_min)
        
        # JSON/structured output detection
        if prompt_snippet.strip().startswith('{') or '"type":' in prompt_snippet:
            # Structured output - keep tight
            return max(self.config.k_min, 6)
        
        return base_k

# Logging integration
def log_adjustment(old_k: int, new_k: int, acceptance: float, reason: str):
    """Log to agent audit log"""
    import json
    from datetime import datetime
    
    entry = {
        "ts": datetime.now().isoformat(),
        "session": "adaptive_spec",
        "level": "INFO", 
        "cat": "SPEC_ADJUST",
        "msg": f"K: {old_k} -> {new_k}",
        "details": f"acceptance={acceptance:.2f}, reason={reason}"
    }
    
    log_path = "/mnt/raid0/llm/LOGS/agent_audit.log"
    with open(log_path, "a") as f:
        f.write(json.dumps(entry) + "\n")
```

### Phase 2: Integration with llama-server

llama-server exposes metrics at `/metrics` endpoint. Parse:
- `llamacpp:draft_accepted_total`
- `llamacpp:draft_drafted_total`

**Wrapper script:** `/mnt/raid0/llm/UTILS/run_adaptive_server.sh`

```bash
#!/bin/bash
# Adaptive speculative decoding server wrapper

source /mnt/raid0/llm/claude/agent_log.sh
agent_session_start "Adaptive speculative decoding experiment"

MODEL_MAIN="/mnt/raid0/llm/models/DeepSeek-R1-32B-Q4_K_M.gguf"
MODEL_DRAFT="/mnt/raid0/llm/models/Qwen2.5-0.5B-Draft-Q4_K_M.gguf"
LLAMA_SERVER="/mnt/raid0/llm/llama.cpp/build/bin/llama-server"

export OMP_NUM_THREADS=1

agent_task_start "Start adaptive inference server" "Testing dynamic K adjustment"

# Start server with initial K=8
numactl --interleave=all $LLAMA_SERVER \
  -m $MODEL_MAIN \
  --draft $MODEL_DRAFT \
  --draft-max 8 \
  -t 96 \
  --mlock \
  --host 0.0.0.0 \
  --port 8080 \
  --metrics &

SERVER_PID=$!
agent_observe "server_pid" "$SERVER_PID"

# Give server time to load
sleep 60

# Run adaptive controller
python3 /mnt/raid0/llm/UTILS/adaptive_speculative.py \
  --server http://localhost:8080 \
  --monitor-interval 10

agent_task_end "Adaptive inference server" "stopped"
```

### Phase 3: Benchmark Comparison

Compare:
1. Fixed K=8 (baseline)
2. Fixed K=16 (aggressive)
3. Adaptive K (this experiment)

Metrics:
- Tokens/second (overall)
- Tokens/second by context type (code, prose, math)
- Acceptance rate distribution

## Success Criteria
- Adaptive K achieves ≥10% higher t/s than best fixed K
- OR: Maintains same t/s with lower variance across context types

## Prerequisites
- [x] GGUF models converted (main + draft) - Qwen2.5-Coder-32B + Qwen2.5-0.5B available
- [x] llama.cpp built with server support - Build 7371 verified
- [x] Baseline benchmarks complete (fixed K) - K=4,8,16 tested, K=8 optimal

## Baseline Results (Qwen2.5-Coder-32B + Qwen2.5-0.5B)
| K | Acceptance | Speed | Notes |
|---|------------|-------|-------|
| 4 | 81.25% | 10.40 t/s | Conservative |
| 8 | 50.00% | 11.01 t/s | **Optimal** |
| 16 | 35.00% | 10.25 t/s | Too aggressive |

## Execution Steps

1. Complete fixed-K benchmarks first (bench_zen5.sh)
2. Create adaptive_speculative.py controller
3. Run llama-server with metrics enabled
4. Run adaptive controller monitoring
5. Compare results

## Token Type Heuristics (Quick Reference)

| Context | Indicators | Recommended K |
|---------|------------|---------------|
| Code | `{ } ( ) ; ``` | 16-24 (high) |
| Prose | Normal text | 8-12 (medium) |
| Math | ∑ ∀ ∫ proof | 4-8 (low) |
| JSON/Schema | `"type":` `{` | 4-6 (very low) |
| Lists/Bullets | `- * 1.` | 8-12 (medium) |

## References
- DISCO: Dynamic Speculative Decoding (2024)
- SpecInfer: Accelerating LLM Serving
- llama.cpp speculative decoding implementation
