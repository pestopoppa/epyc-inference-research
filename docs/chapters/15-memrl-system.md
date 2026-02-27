# Chapter 15: MemRL System

## Introduction

The Memory-augmented Reinforcement Learning (MemRL) system enables the orchestrator to learn from experience. Episodic memory stores task contexts, actions, and outcomes. A two-phase retriever finds similar past situations. Q-value scoring learns which actions work best for which tasks. The system progresses through 7 phases, from manual routing to full learned orchestration.

As of 2026-01-31, the system contains 2,714 memories (1,213 canonical seeds + 1,501 non-seed) with FAISS-accelerated retrieval providing 35x speedup over NumPy baseline (70ms → ~2ms at 500K scale). A contamination cleanup on 2026-01-31 surgically removed 6,506 entries from buggy validation script runs while preserving the original Jan 28 seed corpus.

As of 2026-02-02, the seeding pipeline was enriched with 90 mode-advantage tasks (see [Chapter 24](24-benchmark-suite-construction.md)) specifically designed to produce strong comparative rewards. Prior to this, the debug suite (327 single-turn QA) produced weak routing signal — all tasks were solvable by direct inference, so MemRL learned cost-awareness but not routing quality. The mode-advantage tasks shift +1.0 rewards from ~5% to ~25-35% of episodes. Three external HuggingFace dataset adapters (GAIA 165q, CRUXEval 1600q, BigCodeBench 1140q) further expand the evaluation pool.

## Episodic Memory Architecture

The episodic store is the heart of MemRL — it persists every routing decision the orchestrator makes so it can learn from them later. It uses SQLite for rich metadata queries and FAISS for fast similarity search, living side by side on disk. Note that this store handles *routing memories* only; for codebase retrieval (finding source code and docs), see the NextPLAID integration in [Ch11: REPL Environment](11-repl-environment.md). These are complementary systems: BGE 1024-dim single-vector vs ColBERT 128-dim multi-vector.

<details>
<summary>Storage layout and design rationale</summary>

```
/mnt/raid0/llm/epyc-orchestrator/orchestration/repl_memory/sessions/
├── episodic.db           # SQLite metadata (action, context, q_value, timestamps)
├── embeddings.faiss      # FAISS index (L2-normalized inner product)
└── id_map.npy           # memory_id → faiss_idx mapping
```

**Design Rationale**: SQLite for rich queries (filter by action_type, q_value), FAISS for O(log n) similarity search.

</details>

<details>
<summary>Memory schema and index strategy</summary>

<details>
<summary>SQL: memories table DDL</summary>

```sql
CREATE TABLE memories (
    id TEXT PRIMARY KEY,
    embedding_idx INTEGER NOT NULL,
    action TEXT NOT NULL,
    action_type TEXT NOT NULL,  -- "routing", "escalation", "exploration"
    context TEXT NOT NULL,       -- JSON task context
    outcome TEXT,                -- "success", "failure"
    q_value REAL DEFAULT 0.5,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    update_count INTEGER DEFAULT 0,
    model_id TEXT               -- tracks which model produced this memory (warm-start)
);

CREATE INDEX idx_action_type ON memories(action_type);
CREATE INDEX idx_q_value ON memories(q_value DESC);
CREATE INDEX idx_type_q ON memories(action_type, q_value DESC);
```

</details>

**Indexes Optimized For**:
- Two-phase retrieval (filter by action_type + Q-value)
- Top-k Q-value queries for graph seeding
- Temporal queries (created_at DESC)

</details>

<details>
<summary>FAISS backend implementation and benchmarks</summary>

<details>
<summary>Code: FAISSEmbeddingStore class</summary>

```python
class FAISSEmbeddingStore:
    """FAISS IndexFlatIP with L2 normalization for cosine similarity."""

    def __init__(self, path: Path, dim: int = 1024):
        # BGE-large embedding dim
        self.index = faiss.IndexFlatIP(dim)  # Inner product

    def add(self, memory_id: str, embedding: np.ndarray) -> int:
        # L2 normalize for cosine similarity
        faiss.normalize_L2(embedding)
        self.index.add(embedding)
        self.id_map.append(memory_id)

    def search(self, query: np.ndarray, k: int = 20) -> list[tuple[str, float]]:
        faiss.normalize_L2(query)
        scores, indices = self.index.search(query, k)
        return [(self.id_map[idx], score) for idx, score in zip(indices[0], scores[0])]
```

</details>

**Performance Expectations**:

| Memory Count | FAISS Search Time | NumPy Baseline | Speedup |
|--------------|-------------------|----------------|---------|
| 5K | 0.5ms | 15ms | 30x |
| 50K | 1ms | 150ms | 150x |
| 500K | 2ms | 1500ms | 750x |
| 1M | 3ms | 3000ms | 1000x |

At 2714 memories (current), FAISS overhead is negligible vs NumPy.

</details>

## Task Embedding

Tasks get embedded into 1024-dim vectors so we can find semantically similar past experiences. The `TaskEmbedder` talks to a BGE-large HTTP server for fast embeddings (2-5ms), falling back to subprocess or deterministic hash pseudo-embeddings when the server is unavailable. The serialization focuses on task semantics (type, objective, priority) rather than content, so similar tasks cluster together regardless of their actual input data.

<details>
<summary>TaskEmbedder implementation</summary>

<details>
<summary>Code: TaskEmbedder class and serialization</summary>

```python
class TaskEmbedder:
    """Generate embeddings via HTTP server (2-5ms) or subprocess (50-200ms)."""

    def __init__(self):
        self.model_path = "bge-large-en-v1.5-f16.gguf"
        self.server_url = "http://127.0.0.1:8090"
        self.embedding_dim = 1024  # BGE-large embedding dim

    def embed_task_ir(self, task_ir: Dict[str, Any]) -> np.ndarray:
        # Serialize to focus on semantic fields
        text = self._serialize_task_ir(task_ir)
        return self._generate_embedding(text)
```

```python
def _serialize_task_ir(self, task_ir: Dict[str, Any]) -> str:
    parts = [
        f"type:{task_ir['task_type']}",
        f"objective:{task_ir['objective']}",
        f"priority:{task_ir['priority']}",
        f"constraints:{','.join(task_ir['constraints'][:5])}",
        f"input_types:{','.join(inp['type'] for inp in task_ir['inputs'])}",
    ]
    return " | ".join(parts)
```

</details>

**Why This Format**: Focuses on task semantics, not content. Similar tasks have similar embeddings even with different input data.

</details>

<details>
<summary>Backend fallback chain</summary>

1. **HTTP Server** (8090): 2-5ms via `/embedding` endpoint (40x faster than subprocess)
2. **Subprocess** (`llama-embedding`): 50-200ms with `--embd-output-format json`
3. **Hash-Based Pseudo-Embeddings**: Deterministic SHA-256 expansion (fallback when model unavailable)

Hash fallback preserves identity (same input = same embedding) but NOT similarity. Used only in dev environments without model access.

</details>

## Two-Phase Retrieval

Retrieval works in two passes: first, FAISS finds the nearest neighbors by embedding similarity (casting a wide net with 2x over-fetching), then SQLite filters and ranks them by action_type, Q-value, and a combined score that weights learned utility at 70% and semantic similarity at 30%. This means a high-Q dissimilar memory beats a low-Q similar one — the system trusts what it has learned over surface-level resemblance.

<details>
<summary>Phase 1: semantic filtering implementation</summary>

<details>
<summary>Code: retrieve_by_similarity with SQL enrichment</summary>

```python
def retrieve_by_similarity(
    self,
    query_embedding: np.ndarray,
    k: int = 20,
    action_type: Optional[str] = None,
    min_q_value: float = 0.0,
) -> List[MemoryEntry]:
    # FAISS search (O(log n))
    candidates = self._embedding_store.search(query_embedding, k=k * 2)

    # Phase 2: SQLite filter and enrich
    memory_ids = [memory_id for memory_id, score in candidates]
    placeholders = ",".join("?" * len(memory_ids))
    query = f"""
        SELECT id, embedding_idx, action, action_type, context, outcome, q_value
        FROM memories
        WHERE id IN ({placeholders})
    """
    if action_type:
        query += " AND action_type = ?"
    if min_q_value > 0:
        query += " AND q_value >= ?"

    # Return top k sorted by similarity
```

</details>

**Over-Fetching Strategy**: Retrieve 2k candidates from FAISS, then filter in SQL. Accounts for action_type/q_value filters without separate FAISS indexes.

</details>

<details>
<summary>Phase 2: Q-value ranking and combined scoring</summary>

<details>
<summary>Code: RetrievalConfig dataclass</summary>

```python
@dataclass
class RetrievalConfig:
    semantic_k: int = 20           # Candidates from Phase 1
    min_similarity: float = 0.3    # Cosine similarity threshold
    q_weight: float = 0.7          # Emphasize learned utility
    top_n: int = 5                 # Final results
    confidence_threshold: float = 0.6  # Min combined score to trust
```

</details>

**Combined Score**: `0.7 * q_value + 0.3 * similarity`

Emphasizes learned utility (Q-value) over pure semantic match. A high-Q dissimilar memory is preferred over a low-Q similar one.

</details>

## Q-Value Learning

The system learns which actions work well through TD-learning updates. When a task completes, the QScorer computes a reward based on success/failure, gate failures, and escalation count, then nudges the Q-value toward that reward. Q-values converge over time: 0.9+ means a reliably successful pattern, 0.5 is neutral (the default), and below 0.3 signals likely failure. The QScorer runs asynchronously every 5 minutes so it never blocks inference.

<details>
<summary>TD-learning update and reward computation</summary>

<details>
<summary>Code: Q-value update and reward signal</summary>

```python
def update_q_value(self, memory_id: str, reward: float, learning_rate: float = 0.1) -> float:
    """Q(m) <- Q(m) + alpha(r - Q(m))"""
    old_q = self.get_q_value(memory_id)
    new_q = old_q + learning_rate * (reward - old_q)
    new_q = max(0.0, min(1.0, new_q))  # Clamp to [0, 1]
    self.store.update(memory_id, q_value=new_q, update_count=old_q_count + 1)
    return new_q
```

```python
def _compute_reward(outcome, gate_failures, escalations) -> float:
    if outcome == "success":
        base_reward = 1.0
    elif outcome == "partial":
        base_reward = 0.3
    else:
        base_reward = -0.5

    penalty = gate_failures * 0.1 + escalations * 0.15
    return max(-1.0, min(1.0, base_reward - penalty))
```

</details>

**Interpretation**:
- Q=0.9+: Highly successful pattern
- Q=0.5: Neutral (default)
- Q=0.3-: Likely to fail or escalate

</details>

<details>
<summary>Async QScorer batch processing</summary>

<details>
<summary>Code: QScorer class</summary>

```python
class QScorer:
    """Async Q-value update agent (runs every 5 min)."""

    def score_pending_tasks(self) -> Dict[str, Any]:
        unscored_task_ids = self.reader.get_unscored_tasks()

        for task_id in unscored_task_ids[:batch_size]:
            trajectory = self.reader.get_task_trajectory(task_id)
            reward = self._compute_reward(trajectory)

            # Update routing memory
            routing_memory_id = find_routing_memory(trajectory)
            self.store.update_q_value(routing_memory_id, reward)

            # Update escalation memories
            for escalation in trajectory.escalations:
                self.store.update_q_value(escalation.memory_id, reward)
```

</details>

Keeps Q-updates off the critical inference path. Runs periodically via cron or on-demand trigger.

</details>

<details>
<summary>Multi-dimensional cost model</summary>

QScorer penalizes cost across 3 independent dimensions, each with its own lambda:

<details>
<summary>Code: three-dimensional cost penalty</summary>

```python
# Dimension 1: Latency cost (original)
latency_penalty = cost_penalty_lambda * cost_ratio
# cost_ratio = actual_elapsed / expected_elapsed

# Dimension 2: Quality gap penalty (new)
quality_gap_penalty = cost_lambda_quality_gap * max(0, model_quality - 0.75)
# Applied only when answer is correct. Penalizes using expensive models
# when cheaper ones would suffice.

# Dimension 3: Memory tier penalty (new)
memory_tier_penalty = cost_lambda_memory * (mem_cost - 1.0)
# Applied only for WARM tier models (loaded on demand).
# mem_cost normalized: HOT=1.0, architect_general=3.0, architect_coding=5.0

total_cost_penalty = latency_penalty + quality_gap_penalty + memory_tier_penalty
reward = base_reward - total_cost_penalty
```

</details>

**Quality gap baseline scores** (from benchmark suite, `baseline_quality_by_role`):

| Model | Role | Baseline Quality |
|-------|------|-----------------|
| Qwen3-235B-A22B | architect_general | 0.94 |
| Qwen2.5-Coder-32B | coder | 0.915 |
| Qwen3-Coder-30B-A3B | orchestrator | 0.895 |
| Qwen2.5-7B | worker_explore | 0.745 |

**Interpretation**: If a task is answered correctly by the 235B architect (quality=0.94), dimension 2 penalizes with `lambda * (0.94 - 0.75) = lambda * 0.19`. The same correct answer from 7B (quality=0.745) receives zero quality gap penalty. This teaches the system to prefer cheap models when they can solve the task.

</details>

<details>
<summary>Try-cheap-first Q-value convergence strategy</summary>

The cost model drives a "try cheap first" routing strategy through Q-value convergence:

```
Q(task_class, "worker_explore") learns from:
  - Success → high reward (correct + zero quality gap penalty + HOT tier)
  - Failure → low reward → system escalates to coder/architect
```

During orchestration, Phase B/C nodes check `Q(task_class, "worker_explore") > threshold` to decide whether to attempt the cheap model first. As Q-values converge, the system learns which task classes the 7B worker can handle — routing those directly — and which require immediate escalation, avoiding wasted cheap attempts.

</details>

## MemRL Phases

The system has evolved through 8 phases, from manual YAML-based routing all the way to models making their own routing decisions via REPL tools. All phases are now in production.

<details>
<summary>Phase progression table</summary>

| Phase | Capability | Status (2026-01) |
|-------|------------|------------------|
| 1 | Manual routing via `model_registry.yaml` | Production |
| 2 | Episodic store with embeddings | Production (2714 memories) |
| 3 | Two-phase retrieval (semantic + Q-value) | Production |
| 4 | Learned routing (HybridRouter) | Production |
| 5 | Proactive delegation (complexity-aware) | Production |
| 6 | Graph-enhanced retrieval (failure anti-memory) | Production |
| 7 | FAISS migration (O(log n) embedding search) | Production |
| 8 | Model self-routing (REPL tools + routing context) | Production |

**Current Focus**: Phase 8 (model self-routing) is production-ready. Models can query MemRL Q-values directly via REPL tools and make informed escalation/delegation decisions.

</details>

## MemRL Quality Review Gate

When the MemRL Q-value for a role+task combination drops below 0.6, a two-phase quality review kicks in. First, the architect model gives a quick verdict (OK or WRONG with corrections, ~6s). Only if the verdict is WRONG does a second phase run: the fast worker revises the answer using the architect's corrections (~11s). This triggers on about 20% of requests and adds only ~1.9s average latency — 3x more efficient than routing everything through the architect.

<details>
<summary>Review gate details and performance impact</summary>

**Phase 1 — Architect Verdict** (6.75 t/s, ~40 tokens, ~6s):
- Receives question + answer (TOON-encoded if worker digests available)
- Outputs: `OK` (return unchanged) or `WRONG: <concise corrections>` (trigger Phase 2)

**Phase 2 — Worker Revision** (44 t/s, ~500 tokens, ~11s, only on WRONG):
- Receives: question + original answer + architect corrections
- Outputs: revised answer incorporating corrections

**Performance Impact**:
- Trigger rate: ~20% of requests (Q < 0.6)
- WRONG rate: ~30% of reviews
- Net: ~1.9s average added latency (20% x (6s + 30% x 11s))
- This is 3x more efficient than full architect review (~6s avg vs ~18s)

**Implementation**: `src/api/routes/chat.py` (`_should_review`, `_architect_verdict`, `_fast_revise`)

</details>

## Model Self-Routing (Phase 8)

Models now have agency in routing decisions through 5 REPL functions. On the first turn, compact MemRL Q-values for similar tasks get injected into the routing context (TOON-encoded when there are 2+ results), so models can make informed decisions without even calling the REPL explicitly.

<details>
<summary>REPL routing functions</summary>

| Function | Purpose |
|----------|---------|
| `my_role()` | Self-awareness: role, tier, capabilities |
| `route_advice(task)` | MemRL Q-values + recommended role |
| `delegate(prompt, role, reason)` | Tracked delegation with outcome logging |
| `escalate(reason, target_role)` | Request escalation to specific target |
| `recall(query)` | Episodic memory search with Q-values |

**Routing context** injected on turn 0: compact MemRL Q-values for similar tasks (TOON-encoded when >=2 results). Models use this to make informed routing decisions without explicit REPL calls.

</details>

## Performance Metrics

The system currently holds 2,714 memories split across routing, escalation, and exploration types. End-to-end retrieval (embed + FAISS + SQL) takes 5-13ms, well within interactive latency budgets. The failure and hypothesis graphs track error patterns and action-task confidence, and since the pydantic-graph migration they are fully wired into the orchestration graph nodes rather than sitting as dead code.

<details>
<summary>Memory statistics and graph wiring</summary>

<details>
<summary>Data: memory breakdown (2026-01-28)</summary>

```
Total memories: 2714
├── routing: 1205 (avg Q=0.62)
├── escalation: 892 (avg Q=0.51)
└── exploration: 617 (avg Q=0.68)

Overall avg Q: 0.607
Backend: faiss
Embeddings count: 2714
```

</details>

**Graph Stats** (when enabled):
- Failure graph: Links memories to symptom patterns
- Hypothesis graph: Tracks action-task confidence

**Graph Wiring** (as of 2026-02-07, pydantic-graph migration):
The following MemRL functions are now called from `src/graph/nodes.py`:
- `failure_graph.record_failure()` — called on every error in `_handle_error()`
- `failure_graph.record_mitigation()` — called when an escalated role resolves a failure
- `hypothesis_graph.add_evidence()` — called on task success/failure outcomes
- `retriever.retrieve_for_escalation()` — called during `_check_memrl_suggestion()`

These were previously dead code (declared but never invoked) in the old `repl_executor.py` manual loop.

</details>

<details>
<summary>Retrieval latency breakdown</summary>

| Operation | Time | Notes |
|-----------|------|-------|
| Embed query | 2-5ms | HTTP server (8090) |
| FAISS search (2714 entries) | <1ms | O(log n) |
| SQL filter + enrich | 3-8ms | Indexed queries |
| **Total retrieval** | **5-13ms** | Fast enough for interactive |

With 500K memories (projected), FAISS search would be ~2ms, total ~10-20ms.

</details>

## Replay Evaluation Harness

As of 2026-02-13, we have an offline replay harness that evaluates candidate memory configurations without running live inference. Motivated by ALMA (Xiong et al., Feb 2026), which shows meta-learned memory designs consistently outperform hand-crafted ones. The harness extracts trajectories from progress logs, replays them through isolated episodic stores with different config knobs, and compares routing accuracy, cumulative reward, and Q-convergence speed.

<details>
<summary>Architecture and data flow</summary>

```
ProgressReader → TrajectoryExtractor → [Trajectory]
                                            |
DesignCandidate --> ReplayEngine --> ReplayMetrics
  (RetrievalConfig,    (isolated       (routing_accuracy,
   ScoringConfig)       EpisodicStore)  cumulative_reward,
                                        q_convergence)
                                            |
                                      DesignArchive (SQLite)
                                            |
                                      MetaAgentWorkflow
                                        (Claude proposes ->
                                         replay evaluates ->
                                         human approves)
```

</details>

<details>
<summary>Module inventory and line counts</summary>

| Module | File | LOC | Purpose |
|--------|------|-----|---------|
| Trajectory | `replay/trajectory.py` | 374 | Extract complete trajectories from progress logs, stratified sampling, embedding pre-computation |
| Engine | `replay/engine.py` | 339 | Create isolated EpisodicStore per candidate, replay chronologically, collect per-step results |
| Metrics | `replay/metrics.py` | 102 | Aggregate replay results into comparable metrics (routing accuracy, reward, Q-convergence) |
| Candidates | `replay/candidates.py` | 305 | DesignCandidate (config bundle with lineage) + DesignArchive (SQLite-backed results store) |
| Warm Start | `replay/warm_start.py` | 265 | Model swap detection, Q-value reset, warmup learning rate doubling |
| Meta Agent | `replay/meta_agent.py` | 470 | Claude-as-meta-agent workflow: reflection prompt, candidate parsing, evaluation, promotion recommendation |

**Tests**: 75 tests across 5 files (1,250 LOC). Total production: 1,885 LOC.

</details>

<details>
<summary>Key design decisions</summary>

- **No live embedder calls**: Replay uses pre-computed embeddings from `TrajectoryExtractor`. A `NullEmbedder` safety guard raises if the engine ever tries to call the live embedder.
- **Isolated stores**: Each candidate gets a fresh `EpisodicStore(tmp_dir)` — no cross-contamination between evaluations. Cleaned up after run.
- **No graph integration in v1**: FailureGraph/HypothesisGraph deferred (Kuzu per-candidate too expensive).
- **Human-in-the-loop promotion**: Meta-agent recommends but never auto-promotes. Human reviews markdown report, manually updates `model_registry.yaml`.
- **Stratified sampling**: Default 1000 trajectories, proportional by task_type, reproducible via fixed seed.

</details>

<details>
<summary>model_id field and warm-start support</summary>

Added `model_id TEXT` column to `memories` table (backward-compatible ALTER TABLE, default NULL). Enables:
- **Retrieval affinity**: Same-model memories get +15% score bonus in TwoPhaseRetriever Phase 2
- **Model swap detection**: `WarmStartProtocol.detect_model_swap()` checks if majority of role's memories come from a different model
- **Q-value reset**: On swap, reset Q-values to 0.5 and double learning rate for 50-task warmup period

</details>

<details>
<summary>Baseline replay results (2026-02-13)</summary>

First baseline run against 31 days of progress logs:

| Metric | Value | Notes |
|--------|-------|-------|
| Trajectories extracted | 1160 (1000 sampled) | 160 skipped incomplete |
| Task types | chat, chat_stream | Seeding/eval mock routing |
| Routing accuracy | 0.0% | Expected: all routing_decision="unknown" (mock strategy) |
| Cumulative reward | 997.0 | Nearly all success outcomes |
| Avg reward | 0.997 | |
| Q convergence step | 10 | Quick convergence on homogeneous data |
| Replay duration | 0.18s | 1000 trajectories, no inference |

Routing accuracy will become meaningful once live orchestration produces real routing decisions (not mock/seeding).

</details>

## SkillBank Layer (February 2026)

The episodic store now has a derived knowledge layer called the SkillBank that distills raw trajectories into structured, reusable skills. This matters because SkillRL's ablation data shows -28.2% (ALFWorld) and -22.5% (WebShop) performance when replacing structured skills with raw trajectories. The episodic store remains the ground-truth log; SkillBank is a materialized view optimized for inference-time prompt injection.

<details>
<summary>Relationship to episodic store and storage details</summary>

```
EpisodicStore (raw trajectories)   <- replay harness reads these (unchanged)
       | periodic distillation
SkillBank (structured skills)      <- inference-time retrieval injects into prompts
       | recursive evolution
Refined SkillBank                  <- per-category accuracy monitoring triggers updates
```

The episodic store remains the ground-truth log. SkillBank is a **materialized view** — a lossy compression optimized for inference-time prompt injection. Raw trajectories stay intact for replay evaluation, Q-learning, and audit.

SkillBank uses separate SQLite (`skills.db`) and FAISS indices (`skill_embeddings.faiss`, `skill_id_map.npy`) coexisting alongside the episodic memory files in the same directory.

**Feature Flag**: Gated behind `ORCHESTRATOR_SKILLBANK=1` (requires `memrl`). When disabled, the system operates identically to pre-SkillBank behavior.

**Full documentation**: [Chapter 27: SkillBank & Experience Distillation](27-skillbank-experience-distillation.md)

</details>

## Signal Purity and Robust Confidence (2026-02)

The retrieval layer cleanly separates three concerns: relevance, confidence, and cost. Similarity is used only for neighborhood retrieval. Confidence comes from robust Q statistics (median or trimmed_mean) over top neighbors. Cost is applied at selection time in the `selection_score`, never contaminating the core `P(success|action)` estimate.

<details>
<summary>Calibration and risk metrics</summary>

Additional replay metrics now track calibration/risk:

- `ece_global`
- `brier_global`
- `conformal_coverage`
- `conformal_risk`

Files:

- `orchestration/repl_memory/retriever.py`
- `orchestration/repl_memory/replay/engine.py`
- `orchestration/repl_memory/replay/metrics.py`

</details>

## Regret-Optimized Replay Objective (2026-02)

Replay metrics now include a teacher-match utility objective that combines chosen-pass signal, cost term, and regret penalty. The meta-agent uses `rm_softmax_score` for candidate ranking and promotion instead of raw cumulative reward. Replay action simulation uses action-level posterior scoring (not just top-memory action), which improves sensitivity to retrieval and risk knob changes.

<details>
<summary>Replay metric fields</summary>

- `utility_score`
- `rm_softmax_score`
- `regret_mean`
- `regret_p95`
- `speedup_vs_teacher_mean`
- `route_flip_rate`
- `posterior_margin_mean`

</details>

## GraphRouter Augmentation (February 2026)

The HybridRouter now supports an optional parallel GNN-based routing signal via `GraphRouterPredictor`. This addresses the **cold-start problem**: when a new model joins the fleet, it has zero episodic memories and routing degrades to rule-based fallback.

**How it works**: A bipartite graph (query clusters ↔ LLM roles) is built from episodic memory via MiniBatchKMeans clustering. A 2-layer GAT learns routing patterns from graph structure. For new models, the GAT generalizes from capability embeddings through shared query neighborhoods — no organic data needed.

**Integration**: Blend weight anneals 0.1→0.3 by store size. TwoPhaseRetriever always dominates (70%+ influence). Feature-gated: `ORCHESTRATOR_GRAPH_ROUTER=1`.

See [Chapter 16: Graph-Based Reasoning](16-graph-reasoning.md) for full architecture details.

**Key files**: `routing_graph.py`, `lightweight_gat.py`, `graph_router_predictor.py`, `scripts/graph_router/train_graph_router.py`

## Literature Mapping (Architecture Review Alignment)

This chapter's design choices map directly to the architecture review's research threads. Each theme has a practical interpretation and concrete code anchor.

<details>
<summary>Review theme to code anchor mapping</summary>

| Review Theme | Practical Interpretation | Code Anchors |
|--------------|--------------------------|--------------|
| Regret minimization for routing | Optimize teacher-match utility, not only raw pass rate | `orchestration/repl_memory/replay/engine.py`, `orchestration/repl_memory/replay/meta_agent.py` |
| Cost-aware but signal-pure learning | Keep confidence estimation separate from cost terms; apply cost at decision time | `orchestration/repl_memory/retriever.py` |
| Cache-aware latency modeling | Separate warm/cold behavior and avoid hidden heuristics in confidence estimates | `orchestration/repl_memory/retriever.py` |
| Prior + posterior composition | Treat heuristics as priors that inform, not override, learned evidence | `orchestration/repl_memory/retriever.py`, `src/api/routes/chat_pipeline/routing.py` |
| Replay as design-selection loop | Evaluate candidate routing configs offline before runtime promotion | `orchestration/repl_memory/replay/candidates.py`, `orchestration/repl_memory/replay/meta_agent.py` |

</details>

## References

<details>
<summary>All references (core concepts, implementation, replay harness, related systems)</summary>

### Core Concepts

1. Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press. (TD-learning, Q-values)

2. Johnson, J., Douze, M., & Jegou, H. (2019). *Billion-scale similarity search with GPUs*. IEEE Transactions on Big Data. (FAISS architecture)

### Implementation

3. `orchestration/repl_memory/episodic_store.py`: Memory storage (861 lines)
4. `orchestration/repl_memory/faiss_store.py`: FAISS backend (343 lines)
5. `orchestration/repl_memory/embedder.py`: Task embedding (393 lines)
6. `orchestration/repl_memory/retriever.py`: Two-phase retrieval + HybridRouter + GraphRouter blending
7. `orchestration/repl_memory/q_scorer.py`: Async Q-learning (502 lines)
8. `orchestration/repl_memory/routing_graph.py`: Bipartite routing graph (Kuzu)
9. `orchestration/repl_memory/lightweight_gat.py`: Pure numpy 2-layer GAT
10. `orchestration/repl_memory/graph_router_predictor.py`: Cached GNN inference

### Replay Harness

8. `orchestration/repl_memory/replay/trajectory.py`: Trajectory extraction (374 lines)
9. `orchestration/repl_memory/replay/engine.py`: Offline replay engine (339 lines)
10. `orchestration/repl_memory/replay/candidates.py`: Design candidate archive (305 lines)
11. `orchestration/repl_memory/replay/warm_start.py`: Model swap warm-start (265 lines)
12. `orchestration/repl_memory/replay/meta_agent.py`: Claude meta-agent workflow (470 lines)

### Related Systems

13. Prioritized Experience Replay (Schaul et al., 2015): https://arxiv.org/abs/1511.05952
14. Episodic Memory in Lifelong Learning (Kemker et al., 2018): https://arxiv.org/abs/1802.07569
15. ALMA: Meta-Learned Memory Architectures (Xiong et al., 2026): Motivates offline replay evaluation
16. SkillRL: Evolving Agents via Recursive Skill-Augmented RL (Xia et al., 2026): Motivates SkillBank experience distillation. See [Ch27](27-skillbank-experience-distillation.md).

### Additional Literature (From Architecture Review)

1. Tsiourvas, Sun, Perakis (2025). Causal LLM Routing: End-to-End Regret Minimization from Observational Data. https://openreview.net/forum?id=iZC5xoQQkX
2. Ong et al. (2024). RouteLLM: Learning to Route LLMs with Preference Data. https://arxiv.org/abs/2406.18665
3. DeepSeek-AI (2025). Router-R1: Reinforced Expert Co-Reasoning and Routing. https://openreview.net/forum?id=DWf4vroKWJ
4. Xue et al. (2025). Conformal Risk-Controlled Routing for Large Language Model. https://openreview.net/forum?id=lLR61sHcS5
5. Zheng et al. (2024). SGLang / RadixAttention (cache-aware serving). https://arxiv.org/abs/2312.07104
6. Dai, Yang, Si (2025). S-GRPO: Early Exit via Reinforcement Learning in Reasoning Models. https://arxiv.org/abs/2505.07686

</details>

---

*Previous: [Chapter 14: TOON Encoding](14-toon-encoding.md)* | *Next: [Chapter 16: Graph-Based Reasoning](16-graph-reasoning.md)*
