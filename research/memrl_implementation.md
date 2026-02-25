# MemRL Episodic Memory Implementation

**Status:** Phases 1-7 Complete (FAISS migration done)
**Date:** 2026-01-13 (implementation), 2026-01-14 (lazy loading fix), 2026-01-27 (FAISS migration)
**Paper:** arXiv:2601.03192 - "MemRL: Self-Evolving Agents via Runtime Reinforcement Learning on Episodic Memory" (Zhang et al., 2025)

---

## Overview

Implemented a MemRL-inspired episodic memory system that enables the orchestrator to learn optimal routing, escalation, and exploration strategies from runtime experience without modifying model weights.

**Key insight:** Q-value computation is decoupled from the inference path. A "scorekeeper" monitors progress logs and updates Q-values asynchronously, eliminating latency concerns for interactive routing.

---

## Architecture

```
INFERENCE PATH (synchronous, latency-critical):
  Query → TaskEmbedder → TwoPhaseRetriever → HybridRouter
                              ↓
                    EpisodicStore (pre-scored DB)
                              ↓
                    Routing decision + fallback to rules

LOGGING PATH (lightweight, real-time):
  All Tiers → ProgressLogger → JSONL files (lab book)

SCORING PATH (asynchronous, runs offline):
  ProgressReader → QScorer → EpisodicStore updates
                      ↓
              (Optional) ClaudeAsJudge for graded rewards
```

---

## Components

| Component | File | Purpose |
|-----------|------|---------|
| **EpisodicStore** | `orchestration/repl_memory/episodic_store.py` | SQLite + FAISS/numpy memory storage |
| **FAISSEmbeddingStore** | `orchestration/repl_memory/faiss_store.py` | O(log n) FAISS embedding search |
| **NumpyEmbeddingStore** | `orchestration/repl_memory/faiss_store.py` | O(n) NumPy fallback |
| **TaskEmbedder** | `orchestration/repl_memory/embedder.py` | Task embedding via Qwen2.5-0.5B |
| **TwoPhaseRetriever** | `orchestration/repl_memory/retriever.py` | Semantic + Q-value retrieval |
| **HybridRouter** | `orchestration/repl_memory/retriever.py` | Learned + rule-based routing |
| **ProgressLogger** | `orchestration/repl_memory/progress_logger.py` | Structured JSONL logging |
| **QScorer** | `orchestration/repl_memory/q_scorer.py` | Async Q-value updates |

---

## Two-Phase Retrieval

The retrieval system implements the MemRL paper's two-phase approach:

### Phase 1: Semantic Filtering
- Embed incoming task via TaskEmbedder
- Retrieve top-k candidates by cosine similarity
- Filter by minimum similarity threshold (default: 0.3)

### Phase 2: Q-Value Ranking
- Sort candidates by weighted score: `q_weight * q_value + (1 - q_weight) * similarity`
- Default q_weight: 0.7 (learned utility dominates over semantic similarity)
- Return top-n results with confidence scores

---

## Cold Start Strategy

The system defaults to rule-based routing while the Q-database builds:

| Phase | Timeframe | Behavior |
|-------|-----------|----------|
| Bootstrap | Day 0-30 | System runs normally, Q-scorer observes |
| Hybrid | Day 30-90 | Learned suggestions supplement rules |
| Mature | Day 90+ | Learned routing dominates for common patterns |
| Always | — | Fall back to rules for novel/rare task types |

**Key principle:** No degradation during cold start. The system starts functional and improves over time.

---

## Q-Value Computation

### Basic Mode (Default)
```
reward = base_reward - gate_penalty - escalation_penalty

Where:
- base_reward: success=+1.0, failure=-0.5, partial=+0.3
- gate_penalty: -0.1 per failed gate
- escalation_penalty: -0.15 per escalation
```

### Claude-as-Judge Mode (Optional)
- Use `benchmarks/prompts/v1/orchestrator_planning.yaml` benchmark
- 33 questions across routing, planning, and escalation scenarios
- Graded rewards (0-3 scale) for nuanced learning

---

## Integration Points

### Dispatcher (`src/dispatcher.py`)
- `ProgressLogger` logs routing decisions
- `HybridRouter` provides learned vs rule-based routing
- `routing_strategy` field tracks which method was used

### GateRunner (`src/gate_runner.py`)
- Logs gate pass/fail with task context
- Provides gate failure penalties for Q-scoring

### API (`src/api.py`)
- Real-time Q-scoring after task completion
- Background cleanup during idle periods
- **Lazy loading** (added 2026-01-14): MemRL components only initialize on first `real_mode=True` request

---

## Lazy Loading (Memory Safety)

**Problem:** Eager MemRL initialization loaded TaskEmbedder (0.5B model) on every API startup, causing memory exhaustion during parallel test execution.

**Solution:** Added `_ensure_memrl_initialized()` function that only loads components when needed:
- Mock mode tests never trigger model loading
- Real mode requests initialize on first use
- Background cleanup handles late initialization gracefully

**Files modified:**
- `src/api.py`: Added `_memrl_initialized` flag and lazy init function
- `tests/conftest.py`: Memory guard (100GB threshold)
- `Makefile`: `check-memory` target

---

## Configuration

Added to `orchestration/model_registry.yaml`:

```yaml
repl_memory:
  enabled: true
  database:
    path: /mnt/raid0/llm/claude/orchestration/repl_memory/episodic.db
    embeddings_path: /mnt/raid0/llm/claude/orchestration/repl_memory/embeddings.npy
  embedding:
    model_path: /mnt/raid0/llm/models/Qwen2.5-Coder-0.5B-Instruct-Q8_0.gguf
    dim: 896
    threads: 8
    fallback_enabled: true
  retrieval:
    semantic_k: 20
    min_similarity: 0.3
    q_weight: 0.7
    confidence_threshold: 0.6
  scoring:
    learning_rate: 0.1
    success_reward: 1.0
    failure_reward: -0.5
  cold_start:
    min_samples: 3
    fallback_to_rules: true
    bootstrap_days: 30
```

---

## Completed Phases

### Phase 4-5: Escalation & Exploration Learning (COMPLETE)
See `handoffs/active/memrl-episodic-memory.md` for details.

### Phase 7: FAISS Migration (COMPLETE - 2026-01-27)

Replaced O(n) NumPy mmap with O(log n) FAISS for embedding search.

**Performance improvement:**
| Entries | NumPy (old) | FAISS (new) | Speedup |
|---------|-------------|-------------|---------|
| 5K | ~1ms | ~0.5ms | 2x |
| 50K | ~10ms | ~1ms | 10x |
| 500K | ~70ms | ~2ms | 35x |
| 1M | ~150ms | ~3ms | 50x |

**Files created:**
- `orchestration/repl_memory/faiss_store.py` - FAISS + NumPy backends
- `scripts/migrate_to_faiss.py` - Migration script
- `tests/unit/test_faiss_store.py` - 24 unit tests

**Usage:**
```python
# Default: FAISS backend
store = EpisodicStore(db_path="/path/to/data", use_faiss=True)

# Fallback: NumPy backend
store = EpisodicStore(db_path="/path/to/data", use_faiss=False)
```

---

## Remaining Work

### Phase 6: Claude-as-Judge (Optional)
- [ ] Run orchestrator_planning.yaml benchmark baseline
- [ ] Evaluate graded rewards vs binary
- [ ] Enable if beneficial

---

## Test Commands

```bash
# Verify module imports
python3 -c "from orchestration.repl_memory import EpisodicStore, TaskEmbedder, TwoPhaseRetriever, ProgressLogger, QScorer; print('OK')"

# Check memory stats (after some usage)
python3 -c "from orchestration.repl_memory import EpisodicStore; print(EpisodicStore().get_stats())"

# Run Q-scorer manually
python3 scripts/q_scorer_runner.py --once

# Safe test execution
make check-memory && pytest tests/ -n 4
```

---

## Related Documents

| Document | Purpose |
|----------|---------|
| `handoffs/active/memrl-episodic-memory.md` | Integration checklist |
| `handoffs/active/rlm-orchestrator-roadmap.md` | 8-phase orchestrator roadmap |
| `research/rlm_analysis.md` | RLM paper analysis |
| `orchestration/BLOCKED_TASKS.md` | Phase status tracking |
| `progress/2026-01/2026-01-13.md` | Implementation details |
| `progress/2026-01/2026-01-14.md` | Lazy loading fix |

---

## References

- [MemRL Paper](https://arxiv.org/abs/2601.03192) - Zhang et al., 2025
- [RLM Paper](https://arxiv.org/abs/2512.24601) - Related work on recursive language models
- [research/ESCALATION_FLOW.md](research/ESCALATION_FLOW.md) - Memory pool architecture (HOT/WARM/COLD)
