# Chapter 16: Graph-Based Reasoning

## Introduction

The orchestration system uses three Kuzu graph databases for graph-based reasoning:
- **Failure Graph** (~14MB): Tracks failure modes, symptoms, and mitigations ("anti-memory")
- **Hypothesis Graph** (~4.6MB): Tracks action->task confidence with asymptotic learning
- **Routing Graph** (NEW, Feb 2026): Bipartite (query ↔ LLM) graph for GNN-based cold-start routing

**Key insight:** Current Q-learning optimizes for repeating success. But in debugging and optimization, avoiding known failure modes is often more valuable. And for new model onboarding, graph structure enables inductive generalization without waiting for organic traffic.

All graphs use Kuzu 0.11+ for native graph queries and integrate with the episodic memory store.

## Failure Graph Architecture

The Failure Graph is the system's anti-memory -- it remembers what went wrong so you don't repeat it. It tracks three node types (failure modes, symptoms, mitigations) connected by five relationship types, forming a queryable web of past mistakes and their fixes. When a new error appears, you match its symptoms against known failures and retrieve proven mitigations ranked by success rate.

<details>
<summary>Failure Graph schema and operations</summary>

### Schema

The Failure Graph tracks three node types and five relationship types:

| Node Type | Primary Key | Properties |
|-----------|-------------|------------|
| `FailureMode` | id (STRING) | description, severity (1-5), first_seen, last_seen |
| `Symptom` | id (STRING) | pattern (regex/keyword), detection_method |
| `Mitigation` | id (STRING) | action, success_rate (0.0-1.0), attempt_count, success_count |
| `MemoryLink` | id (STRING) | memory_id (episodic store reference) |

| Relationship | From -> To | Meaning |
|--------------|-----------|---------|
| `HAS_SYMPTOM` | FailureMode -> Symptom | Observable pattern indicating failure |
| `MITIGATED_BY` | FailureMode -> Mitigation | Action that resolved failure |
| `PRECEDED_BY` | FailureMode -> FailureMode | Causal failure chain |
| `RECURRED_AFTER` | FailureMode -> Mitigation | Mitigation didn't work |
| `TRIGGERED_FROM` | MemoryLink -> FailureMode | Links to episodic memory |

### Core Operations

<details>
<summary>Code: recording failures and mitigations</summary>

**Recording a Failure:**

```python
from orchestration.repl_memory.failure_graph import FailureGraph

graph = FailureGraph()

# Detect symptoms from error logs
symptoms = [
    "garbage output",
    "ssm state corruption",
    "model hangs"
]

# Record failure with severity 5 (critical)
failure_id = graph.record_failure(
    memory_id="episode_uuid",
    symptoms=symptoms,
    description="Qwen3-Next state corruption with speculation",
    severity=5,
    previous_failure_id=None  # Optional for causal chains
)
```

**Recording a Mitigation:**

```python
# Try a mitigation
action = "use expert reduction only, no speculation"

# Record outcome
mitigation_id = graph.record_mitigation(
    failure_id=failure_id,
    action=action,
    worked=True  # Success or failure
)
# Updates success_rate automatically based on attempt history
```

</details>

<details>
<summary>Code: querying failures and assessing risk</summary>

**Finding Matching Failures:**

```python
# Current task is failing - find similar failures
current_symptoms = ["garbage output", "repetitive loops"]

failures = graph.find_matching_failures(current_symptoms)
# Returns List[FailureMode] sorted by symptom overlap

if failures:
    # Get effective mitigations
    mitigations = graph.get_effective_mitigations(current_symptoms)
    # Returns [{"action": str, "success_rate": float}]

    best_mitigation = mitigations[0]["action"]
    print(f"Suggested fix: {best_mitigation}")
```

**Failure Risk Assessment:**

```python
# Before executing an action, check failure risk
action = "use speculative decoding with SSM model"
risk = graph.get_failure_risk(action)

if risk > 0.5:
    print(f"WARNING: High failure risk ({risk:.2f}) for this action")
```

</details>

### Seeded Failure Modes

The system is pre-seeded with 14 known failure modes from `orchestration/repl_memory/graph_seeds.yaml`:

| Failure ID | Description | Severity | Symptoms |
|------------|-------------|----------|----------|
| `qwen3_next_ssm` | SSM state corruption with speculation | 5 | garbage output, repetitive loops |
| `qwen3_coder_480b_bos` | BOS token mismatch breaks speculation | 4 | 0% acceptance rate |
| `deepseek_r1_vocab` | Vocab size mismatch (152064 vs 151936) | 3 | token mismatch error |
| `llama_lookup_large_context` | llama-lookup crashes on prompts >10K | 3 | assertion failed n_tokens <= n_batch |
| `vl_models_speculation` | VL models timeout with spec decode | 4 | timeout, mmproj not loaded |
| `interactive_mode_hang` | llama-cli hangs waiting for input | 2 | benchmark hangs |
| `moe_under_4_experts` | MoE models crash with <4 experts | 5 | SIGSEGV, garbage output |
| `repl_tool_noncompliance` | Models use Python imports instead of tools | 3 | security error, import blocked |

</details>

## Hypothesis Graph Architecture

The Hypothesis Graph tracks how confident the system is about specific action-task combinations. Every time an action succeeds or fails on a task type, confidence updates asymptotically -- approaching 1.0 on repeated success and 0.0 on repeated failure. Before the orchestrator suggests an action, it checks this graph and emits warnings when confidence is low.

<details>
<summary>Hypothesis Graph schema and confidence model</summary>

### Confidence Tracking

The Hypothesis Graph tracks confidence in action-task combinations using asymptotic Bayesian updates:

```python
# Learning rate for confidence updates
LEARNING_RATE = 0.1

# On success:
new_confidence = old_confidence + LEARNING_RATE * (1.0 - old_confidence)

# On failure:
new_confidence = old_confidence - LEARNING_RATE * old_confidence
```

This formula ensures confidence asymptotically approaches 1.0 on repeated success and 0.0 on repeated failure.

### Schema

| Node Type | Properties |
|-----------|------------|
| `Hypothesis` | id, claim (action\|task_type), confidence (0.0-1.0), created_at, tested |
| `HypothesisEvidence` | id, evidence_type (supports/contradicts), source, timestamp |
| `HypothesisMemoryLink` | id, memory_id (episodic store reference) |

| Relationship | Meaning |
|--------------|---------|
| `SUPPORTS` | Evidence -> Hypothesis (successful outcome) |
| `CONTRADICTS` | Evidence -> Hypothesis (failed outcome) |
| `GENERATED_FROM` | Hypothesis -> MemoryLink (created from episode) |

### Usage Patterns

<details>
<summary>Code: creating hypotheses and adding evidence</summary>

**Creating Hypotheses:**

```python
from orchestration.repl_memory.hypothesis_graph import HypothesisGraph

graph = HypothesisGraph()

# Create hypothesis for action-task combination
hypothesis_id = graph.get_or_create_hypothesis(
    action="speculative_decode",
    task_type="code_generation",
    memory_id="episode_uuid"
)
```

**Adding Evidence:**

```python
# After executing the action
new_confidence = graph.add_evidence(
    hypothesis_id=hypothesis_id,
    outcome="success",  # or "failure"
    source="benchmark_run_uuid"
)
# Returns updated confidence score
```

</details>

<details>
<summary>Code: pre-execution confidence warnings</summary>

**Pre-execution Warnings:**

```python
# Before suggesting an action
confidence = graph.get_confidence(
    action="speculative_decode",
    task_type="vision_tasks"
)

if confidence < 0.3:
    warnings = graph.get_low_confidence_warnings(
        action="speculative_decode",
        task_type="vision_tasks",
        threshold=0.3
    )
    print("\n".join(warnings))
    # "Low confidence (0.15) for 'speculative_decode' on 'vision_tasks'.
    #  Evidence: benchmark_xyz, episode_abc"
```

</details>

### Seeded Hypotheses

The system is pre-seeded with 15 high-confidence hypotheses from successful benchmarks:

| Hypothesis | Confidence | Rationale |
|------------|------------|-----------|
| `speculative_decode\|code_generation` | 0.85 | Qwen2.5-Coder-32B + 0.5B = 11x speedup |
| `prompt_lookup\|summarization` | 0.82 | 95.18 t/s, 12.7x speedup |
| `expert_reduction_4\|moe_reasoning` | 0.75 | Qwen3-235B at 6.75 t/s with quality |
| `no_speculation\|ssm_models` | 0.95 | Qwen3-Next MUST use expert reduction only |
| `temperature_0.7\|vision_tasks` | 0.70 | Qwen2.5-VL-7B: 28.3 -> 57.1 t/s |

</details>

## Query Patterns

Both graphs support Cypher queries for advanced traversal. You use these when the Python API doesn't cover your specific lookup -- for example, walking causal failure chains or finding untested hypotheses above a confidence threshold.

<details>
<summary>Cypher query examples</summary>

**Get Causal Failure Chain:**

<details>
<summary>Code: failure chain traversal</summary>

```cypher
MATCH path = (f1:FailureMode {id: $id})-[:PRECEDED_BY*1..5]->(f2:FailureMode)
RETURN f2.id, f2.description, f2.severity
ORDER BY f2.first_seen ASC
```

</details>

**Find Untested High-Confidence Hypotheses:**

<details>
<summary>Code: untested hypothesis query</summary>

```cypher
MATCH (h:Hypothesis)
WHERE h.tested = false AND h.confidence >= 0.7
RETURN h.id, h.claim, h.confidence
ORDER BY h.confidence DESC
```

</details>

**Get Supporting Evidence for Hypothesis:**

<details>
<summary>Code: evidence retrieval query</summary>

```cypher
MATCH (e:HypothesisEvidence)-[:SUPPORTS]->(h:Hypothesis {id: $id})
RETURN e.id, e.source, e.timestamp
ORDER BY e.timestamp DESC
```

</details>

</details>

## Storage and Performance

Both graph databases are compact and fast. The Failure Graph holds around 74 nodes across 14MB, the Hypothesis Graph around 15 hypotheses plus evidence in 4.6MB. Both run on Kuzu 0.11+ and live on the RAID array.

<details>
<summary>Storage metrics</summary>

| Metric | Failure Graph | Hypothesis Graph |
|--------|---------------|------------------|
| Database size | ~14MB | ~4.6MB |
| Node count | 13 failures + 45 symptoms + 16 mitigations | 15 hypotheses + evidence |
| Storage location | `/mnt/raid0/llm/epyc-orchestrator/orchestration/repl_memory/kuzu_db/failure_graph/` | `.../hypothesis_graph/` |
| Backend | Kuzu 0.11+ | Kuzu 0.11+ |

</details>

## JSON Canvas Export

Both graphs can be exported to JSON Canvas format for visualization in Obsidian. This lets you see failure patterns and hypothesis confidence as a spatial graph, edit node positions to steer attention, and re-import changes as planning constraints.

<details>
<summary>Canvas export and import workflow</summary>

### Export Functions

<details>
<summary>Code: export functions</summary>

```python
from src.canvas_export import export_hypothesis_graph, export_failure_graph, export_session_context

# Export hypothesis graph
canvas = export_hypothesis_graph(graph, output_path="logs/canvases/hypothesis.canvas")

# Export failure graph
canvas = export_failure_graph(graph, output_path="logs/canvases/failure.canvas")

# Combined session view
canvas = export_session_context(hyp_graph, fail_graph, output_path="logs/canvases/session.canvas")
```

</details>

### Visual Encoding

| Feature | Hypothesis Graph | Failure Graph |
|---------|------------------|---------------|
| Node color | Confidence-based (green >0.7, yellow 0.3-0.7, red <0.3) | Severity-based |
| Evidence nodes | Blue (supports) / Red (contradicts) | Orange (symptoms) / Purple (mitigations) |
| Layout | Grid with priority ordering | Grid with symptoms on left, mitigations on right |

### MCP Tools

Available via MCP server:
- `export_reasoning_canvas(graph_type="hypothesis|failure|session")`
- `import_canvas_edits(canvas_path, baseline_path)` -- extract constraints from edited canvas
- `list_canvases()` -- list available canvas files

### Canvas Import

When a user edits a canvas in Obsidian, the changes can be re-imported as planning constraints:

<details>
<summary>Code: canvas import and LLM context</summary>

```python
from src.canvas_import import load_canvas_for_llm

# Load edited canvas and extract constraints for LLM context
context = load_canvas_for_llm("logs/canvases/hypothesis.canvas", use_toon=True)
# Returns TOON-encoded string with position-based priorities
```

Position-based priority: nodes closer to canvas center are weighted higher, enabling human-in-the-loop attention steering.

</details>

</details>

## Failure Lesson Formalization (February 2026)

The FailureBridge connects the Kuzu-backed FailureGraph with the SkillBank, turning proven mitigations into reusable skills. When a mitigation achieves a high enough success rate with sufficient evidence, it gets promoted from anti-memory into positive structured knowledge that the distillation pipeline can use directly.

<details>
<summary>Bridge operations and qualification criteria</summary>

### Bridge Operations

| Operation | Direction | Purpose |
|-----------|-----------|---------|
| `sync_mitigations_to_skills()` | FailureGraph -> SkillBank | Export mitigations with success_rate >= 0.7 as skills |
| `get_failure_context_for_distillation()` | FailureGraph -> Distiller | Enrich distillation prompts with failure history |
| `check_skill_against_graph()` | SkillBank -> FailureGraph | Cross-reference proposed skills against known failures |

### Qualification Criteria

Only high-quality mitigations are promoted to skills:
- `success_rate >= 0.7` -- proven effective
- `attempt_count >= 3` -- sufficient evidence
- Confidence capped at `min(success_rate, 0.85)` for bridge-generated skills

This extends the Failure Graph from a read-only anti-memory into a source of positive structured knowledge. See [Chapter 27](27-skillbank-experience-distillation.md) for full details.

</details>

## Routing Graph (GraphRouter)

The Routing Graph is a bipartite heterogeneous graph that powers GNN-based cold-start routing. When a new model joins the fleet, it has zero episodic memories. The GAT propagates routing predictions through shared query neighborhoods — no cold-start wait required.

Based on GraphRouter (ICLR 2025, arXiv:2410.03834).

<details>
<summary>Routing Graph schema and architecture</summary>

### Schema

| Node Type | Primary Key | Properties |
|-----------|-------------|------------|
| `TaskType` | id (STRING) | description, embedding (1024-dim BGE-large) |
| `QueryCluster` | id (STRING) | representative_text, embedding, task_type_id, sample_count |
| `LLMRole` | id (STRING) | description, embedding, port, tokens_per_second, memory_tier, memory_gb |

| Relationship | From -> To | Properties |
|--------------|-----------|------------|
| `BELONGS_TO` | QueryCluster -> TaskType | (none) |
| `PERFORMANCE_ON` | LLMRole -> QueryCluster | success_rate, avg_q_value, avg_latency_s, sample_count, last_updated |

### Architecture

```
BipartiteRoutingGraph (Kuzu)
  ├── sync_from_episodic_store() → MiniBatchKMeans → QueryCluster centroids
  └── get_node_features() + get_edge_index() → GAT input

LightweightGAT (pure numpy, 2-layer)
  ├── Layer 1: MultiHeadGAT(1024→32, heads=4) + ELU → 128-dim
  ├── Layer 2: MultiHeadGAT(128→32, heads=1) + ELU → 32-dim
  └── Edge prediction: sigmoid(dot(query_emb, llm_emb))

GraphRouterPredictor (cached inference)
  ├── Find nearest QueryCluster by cosine sim
  ├── GAT forward on precomputed graph (cached 60s)
  └── Softmax over LLM role scores

HybridRouter (blending)
  └── posterior = (1-w) × retriever_score + w × graph_score
      where w anneals 0.1→0.3 by episodic store size
```

### Cold-Start Onboarding

New models get routing predictions inductively:

```bash
python3 scripts/graph_router/onboard_model.py \
    --role new_coder_v2 \
    --description "Qwen4-Coder-32B, 55 t/s" \
    --port 8086 --tps 55.0 --memory-tier HOT --memory-gb 20
```

The GAT generalizes from the capability embedding through shared query neighborhoods. Hours of organic accumulation → minutes.

### Training

```bash
python3 scripts/graph_router/train_graph_router.py --epochs 100 --lr 0.001
```

Edge masking (20% held out) → BCE loss → SGD with cosine LR decay. Runtime: ~10-30s on CPU.

</details>

<details>
<summary>References</summary>

- **Source**: `orchestration/repl_memory/failure_graph.py`, `hypothesis_graph.py`, `routing_graph.py`
- **GAT**: `orchestration/repl_memory/lightweight_gat.py`
- **Predictor**: `orchestration/repl_memory/graph_router_predictor.py`
- **Failure Bridge**: `orchestration/repl_memory/distillation/failure_bridge.py`
- **Seeds**: `orchestration/repl_memory/graph_seeds.yaml`
- **Training**: `scripts/graph_router/train_graph_router.py`, `onboard_model.py`
- **Model quirks**: Extracted from `docs/reference/models/QUIRKS.md`
- **Benchmark evidence**: Extracted from `benchmarks/results/reviews/summary.csv`
- **Paper**: GraphRouter (ICLR 2025, arXiv:2410.03834)
- **Benchmark**: LLMRouterBench (arXiv:2601.07206) — routing gains are coarse-grained

</details>

---

*Previous: [Chapter 15: MemRL System](15-memrl-system.md)* | *Next: [Chapter 17: Memory Seeding](17-memory-seeding.md)*
