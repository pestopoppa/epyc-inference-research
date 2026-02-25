# Chapter 27: SkillBank & Experience Distillation

## Introduction

Raw episodic trajectories are redundant, noisy, and scale poorly as prompt context. SkillRL's ablation data quantifies this: replacing structured skills with raw trajectories causes **-28.2%** (ALFWorld) and **-22.5%** (WebShop) performance drops. SkillBank addresses this by distilling raw trajectories into structured, reusable "skills" — compressed behavioral principles optimized for prompt injection.

**Key insight**: SkillBank is a **materialized view** over the episodic store. Raw trajectories remain intact for replay evaluation, Q-learning, and audit. Skills are derived, lossy compressions optimized for inference-time retrieval and prompt injection.

**Expected impact**:
- 10-20x token compression per retrieved memory (structured principle vs raw trajectory)
- Monotonic escalation reduction: skills from architect-solved tasks propagate to workers
- Multiplicative gain with replay harness: replay optimizes retrieval params, SkillBank improves data quality — orthogonal axes

SkillRL demonstrates that a 7B model with skill augmentation (89.9% ALFWorld) outperforms GPT-4o (48.0%) and Gemini-2.5-Pro (60.3%). This validates investing in memory quality over model size — directly relevant since our escalation pipeline routes from 7B workers (port 8082, 44 t/s) up to 235B/480B architects.

## Architecture Overview

The SkillBank system sits between the episodic store and the inference pipeline. Teacher models distill raw trajectories into structured skills, which live in their own SQLite + FAISS store. At inference time, a two-level retriever pulls relevant skills and injects them into prompts. The replay engine evaluates skill effectiveness offline, while the episodic store remains the source of truth for raw data.

<details>
<summary>Architecture and data flow</summary>

```
                        ┌─────────────────────────┐
                        │   Teacher Models         │
                        │  ┌───────────────────┐   │
                        │  │ Claude Opus 4.6    │   │
                        │  │ Codex gpt-5.3      │   │
                        │  │ Qwen3-235B (local) │   │
                        │  └───────────────────┘   │
                        └──────────┬──────────────┘
                                   │ distillation
                                   ▼
┌──────────────┐    read     ┌──────────────┐    embed     ┌──────────────┐
│ EpisodicStore│────────────▶│ Distiller    │─────────────▶│ SkillBank    │
│  (episodic.db│             │  Pipeline    │              │  (skills.db  │
│   + FAISS)   │             └──────────────┘              │   + FAISS)   │
└──────┬───────┘                                           └──────┬───────┘
       │                                                          │
       │  replay harness reads                  retriever reads   │
       ▼                                                          ▼
┌──────────────┐                               ┌──────────────────┐
│ ReplayEngine │                               │ SkillRetriever   │
│ (offline     │                               │ (runtime, injects│
│  evaluation) │                               │  into prompts)   │
└──────────────┘                               └────────┬─────────┘
                                                        │
                                               ┌────────▼─────────┐
                                               │ HybridRouter     │
                                               │ (wrapped by      │
                                               │  SkillAugmented- │
                                               │  Router)         │
                                               └──────────────────┘
```

### Separation of Concerns

| Component | Reads | Writes | Frequency |
|-----------|-------|--------|-----------|
| EpisodicStore | — | Raw trajectories | Every task (real-time) |
| Distiller | EpisodicStore, FailureGraph | SkillBank | Periodic batch (daily/weekly) |
| SkillBank | — | Skills (from Distiller) | Batch only |
| SkillRetriever | SkillBank FAISS index | — | Every inference request |
| ReplayEngine | EpisodicStore (raw) | DesignArchive | On-demand (meta-agent) |
| EvolutionMonitor | SkillBank, OutcomeTracker | SkillBank (updates) | Triggered by accuracy drop |

</details>

## SkillBank Core

Each skill is a structured record stored in SQLite with a companion FAISS vector index for similarity search. Skills carry provenance links back to source trajectories, effectiveness tracking fields, and optional failure-specific metadata. The FAISS index for skills coexists alongside the episodic memory's FAISS index using custom filename parameters, and a hard cap of 500 skills prevents unbounded growth.

<details>
<summary>Schema and storage design</summary>

### Schema

Skills are stored in SQLite with FAISS vector search on a separate index (`skill_embeddings.faiss` + `skill_id_map.npy`) coexisting alongside the episodic memory's FAISS index.

<details>
<summary>Code: Skill dataclass</summary>

```python
@dataclass
class Skill:
    id: str                          # "sk_{type}_{uuid[:8]}"
    title: str                       # "Run Tests Before Deploying"
    skill_type: str                  # general | routing | escalation | failure_lesson
    principle: str                   # Core behavioral principle
    when_to_apply: str               # Trigger condition
    task_types: List[str]            # ["code_generation", "debugging"] or ["*"]
    source_trajectory_ids: List[str] # Provenance links to episodic store
    source_outcome: str              # "success" | "failure" | "escalation"

    # Effectiveness tracking
    confidence: float = 0.5          # [0.0, 1.0], evolves via EvolutionMonitor
    retrieval_count: int = 0
    effectiveness_score: float = 0.0 # Rolling average from OutcomeTracker
    deprecated: bool = False

    # Optional fields
    embedding: Optional[np.ndarray] = None
    flawed_reasoning: Optional[str] = None   # For failure_lesson type
    prevention_principle: Optional[str] = None
```

</details>

<details>
<summary>SQL: skills table DDL</summary>

```sql
CREATE TABLE skills (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    skill_type TEXT NOT NULL,
    principle TEXT NOT NULL,
    when_to_apply TEXT NOT NULL,
    task_types TEXT NOT NULL,          -- JSON array
    source_trajectory_ids TEXT NOT NULL, -- JSON array
    source_outcome TEXT NOT NULL,
    confidence REAL DEFAULT 0.5,
    retrieval_count INTEGER DEFAULT 0,
    effectiveness_score REAL DEFAULT 0.0,
    deprecated INTEGER DEFAULT 0,
    flawed_reasoning TEXT,
    prevention_principle TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX idx_skill_type ON skills(skill_type);
CREATE INDEX idx_deprecated ON skills(deprecated);
CREATE INDEX idx_confidence ON skills(confidence DESC);
```

</details>

### Storage Layout

```
/mnt/raid0/llm/claude/orchestration/repl_memory/sessions/
├── episodic.db             # SQLite metadata (existing)
├── embeddings.faiss        # Episodic FAISS index (existing)
├── id_map.npy              # Episodic ID map (existing)
├── skills.db               # SkillBank SQLite (NEW)
├── skill_embeddings.faiss  # Skill FAISS index (NEW)
└── skill_id_map.npy        # Skill ID map (NEW)
```

The FAISS coexistence is enabled by custom filename parameters on `FAISSEmbeddingStore`:

<details>
<summary>Code: FAISS filename configuration</summary>

```python
# Default (episodic)
FAISSEmbeddingStore(path=faiss_dir, dim=1024)
# → embeddings.faiss, id_map.npy

# Skill-specific
FAISSEmbeddingStore(
    path=faiss_dir, dim=1024,
    index_filename="skill_embeddings.faiss",
    id_map_filename="skill_id_map.npy",
)
```

</details>

### Capacity Limits

- `MAX_SKILLS = 500` — hard limit preventing unbounded growth
- `WARN_SKILLS = 400` — log warning when approaching limit
- Skills above limit are handled by EvolutionMonitor deprecation

### Skill Types

| Type | Source | Purpose |
|------|--------|---------|
| `general` | Success trajectories | Always-applicable behavioral principles |
| `routing` | Routing decisions | "For X tasks, prefer role Y because Z" |
| `escalation` | Escalation events | "When seeing symptoms X, escalate to Y" |
| `failure_lesson` | Failure trajectories | "Approach X fails for Y because Z; try W instead" |

</details>

## Two-Level Retrieval

Skill retrieval uses a two-level strategy adapted from SkillRL's Adaptive Skill Retrieval. Level 1 always injects high-confidence general skills regardless of task type. Level 2 uses FAISS cosine similarity to find task-specific skills matching the current request. The `SkillAugmentedRouter` wraps the existing `HybridRouter` transparently, so routing works identically with or without skills enabled.

<details>
<summary>Retrieval levels and prompt injection</summary>

### Level 1: General Skills

Always included in prompt regardless of task type. Filtered by `skill_type = "general"` and `task_types = ["*"]`, sorted by confidence descending.

Default: top 6 general skills (configurable via `SkillBankConfig.general_skills_max`).

### Level 2: Task-Specific Skills

Retrieved by FAISS cosine similarity between task embedding and skill embeddings. Filtered by `min_similarity >= 0.4` and `min_confidence >= 0.3`.

Default: top 6 task-specific skills (configurable via `SkillBankConfig.task_specific_k`).

### Deduplication

Skills appearing in both levels are deduplicated — general skills take priority.

### Prompt Formatting

Skills are formatted as a markdown section injected before the task prompt:

<details>
<summary>Code: prompt injection format</summary>

```markdown
## Relevant Skills

### General Skills
1. **Run Tests Before Deploying** (confidence: 0.92)
   - Principle: Always execute the test suite before any deployment action.
   - When to apply: always

### Task-Specific Skills (code_generation)
1. **Use Type Hints for Complex Returns** (confidence: 0.85, similarity: 0.78)
   - Principle: Annotate return types for functions with non-obvious return structures.
   - When to apply: When implementing functions with complex data structures.
```

</details>

Token budget: `SkillBankConfig.max_prompt_tokens = 1500` (configurable). Skills are added in priority order until the budget is exhausted.

### SkillAugmentedRouter

The `SkillAugmentedRouter` wraps `HybridRouter` transparently:

<details>
<summary>Code: SkillAugmentedRouter</summary>

```python
class SkillAugmentedRouter:
    def route(self, task_ir):
        # Delegates to HybridRouter (backward-compatible)
        return self.hybrid_router.route(task_ir)

    def route_with_skills(self, task_ir):
        routing_decision, strategy = self.hybrid_router.route(task_ir)
        # Retrieve skills, format for prompt injection
        embedding = self.embedder.embed_task_ir(task_ir)
        results = self.skill_retriever.retrieve_for_task(embedding, task_type)
        skill_context = self.skill_retriever.format_for_prompt(results)
        return routing_decision, strategy, skill_context
```

</details>

Skill context is injected into `direct_stage.py` and `repl_executor.py` by prepending to the prompt.

</details>

## Distillation Pipeline

Three teacher models -- Claude Opus 4.6, gpt-5.3-codex, and Qwen3-235B locally -- distill raw trajectories into structured skills. Each teacher implements a simple `distill(prompt) -> str` protocol, and the pipeline batches trajectories, builds type-specific prompts (success, failure, or escalation), and deduplicates new skills against the existing bank using cosine similarity with a 0.85 threshold.

<details>
<summary>Teacher models and pipeline operation</summary>

### Teacher Models

Three teachers provide redundancy and diversity:

| Teacher | Model | Access | Strengths |
|---------|-------|--------|-----------|
| `ClaudeTeacher` | Claude Opus 4.6 | Anthropic API | Deep reasoning, nuanced analysis |
| `CodexTeacher` | gpt-5.3-codex | `codex exec --json` CLI | Code-specific expertise |
| `LocalLlamaTeacher` | Qwen3-235B-A22B | HTTP :8083 | Zero-cost, low-latency, local |

All teachers implement the `TeacherModel` protocol:

<details>
<summary>Code: TeacherModel protocol</summary>

```python
class TeacherModel(Protocol):
    async def distill(self, prompt: str) -> str: ...
    @property
    def name(self) -> str: ...
```

`MockTeacher` is provided for testing (no inference, configurable responses).

</details>

### Distillation Prompts

Three prompt templates for different trajectory types:

**Success distillation** (`build_success_distillation_prompt`):
- Input: task objective, action taken, tools used, outcome details
- Output: structured skill with title, principle, when_to_apply, task_types

**Failure distillation** (`build_failure_distillation_prompt`):
- Input: task, failed action, error details, what went wrong
- Output: failure_lesson skill with flawed_reasoning, prevention_principle

**Escalation distillation** (`build_escalation_distillation_prompt`):
- Input: original role, target role, escalation reason, symptoms observed
- Output: escalation skill with routing guidance

### Pipeline Operation

<details>
<summary>Code: DistillationPipeline.run</summary>

```python
class DistillationPipeline:
    async def run(self, trajectories, batch_size=10) -> DistillationReport:
        for batch in batched(trajectories, batch_size):
            for traj in batch:
                prompt = self._build_prompt(traj)
                response = await self.teacher.distill(prompt)
                skills = parse_skills_from_response(response, traj)

                for skill in skills:
                    if not self._is_duplicate(skill):  # cosine > 0.85 = dup
                        self.skill_bank.store(skill)
```

</details>

Deduplication: new skills with cosine similarity > 0.85 to existing skills are skipped.

</details>

## Failure Lesson Formalization

The `FailureBridge` connects the Kuzu-backed FailureGraph (anti-memory) with the SQLite-backed SkillBank. It exports high-quality mitigations as `failure_lesson` skills, enriches distillation prompts with failure history, and cross-references proposed skills against known failure modes. Only mitigations with a success rate of 0.7 or higher and at least 3 attempts qualify for export.

<details>
<summary>FailureBridge design and enrichment flow</summary>

### FailureBridge

Connects the Kuzu-backed FailureGraph (anti-memory) with the SQLite-backed SkillBank:

<details>
<summary>Code: FailureBridge class</summary>

```python
class FailureBridge:
    def get_failure_context_for_distillation(self) -> str:
        # Generates markdown summary of failure modes, symptoms,
        # and effective mitigations for distillation prompt enrichment

    def sync_mitigations_to_skills(self) -> int:
        # Exports high-quality mitigations (success_rate >= 0.7)
        # as failure_lesson skills in SkillBank

    def check_skill_against_graph(self, skill) -> dict:
        # Cross-references proposed skills against known failure modes
        # Returns match info with confidence adjustment
```

</details>

### Enrichment Flow

```
FailureGraph (Kuzu)  ──sync──▶  SkillBank failure_lessons
       │                              │
       │ symptoms/mitigations         │ structured skills
       ▼                              ▼
Distillation prompts ◀──context──  SkillRetriever
  (enriched with                    (injects into prompts)
   failure history)
```

Mitigations with `success_rate >= 0.7` and `attempt_count >= 3` are exported. Confidence is capped at `min(success_rate, 0.85)` for bridge-generated skills.

</details>

## Recursive Skill Evolution

Adapted from SkillRL's Recursive Skill Evolution, the `EvolutionMonitor` periodically evaluates all active skills and adjusts their confidence based on observed effectiveness. Skills that consistently help get promoted; skills that stop helping decay and eventually get deprecated. The `OutcomeTracker` records task outcomes correlated with skill retrievals, providing the effectiveness signal that drives evolution.

<details>
<summary>Evolution cycle and outcome tracking</summary>

### EvolutionMonitor

Evaluates all active skills periodically and adjusts confidence based on observed effectiveness:

<details>
<summary>Config: EvolutionConfig</summary>

```python
class EvolutionConfig:
    promotion_threshold: float = 0.8    # Effectiveness above this → promote
    deprecation_threshold: float = 0.3  # Confidence below this → deprecate
    min_retrievals: int = 5             # Skip under-retrieved skills
    decay_rate: float = 0.05            # Confidence decay per cycle
    promotion_boost: float = 0.1        # Confidence increase on promotion
    max_confidence: float = 0.95        # Hard ceiling
    stale_days: int = 30                # Mark stale after N days unused
```

</details>

### Evolution Cycle

For each active skill with `retrieval_count >= min_retrievals`:

1. **Compute effectiveness**: From `OutcomeTracker` (task outcomes correlated with skill retrievals) or heuristic fallback (confidence x retrieval frequency)
2. **Promote** (effectiveness >= 0.8): Boost confidence by `promotion_boost`, capped at `max_confidence`
3. **Decay** (effectiveness < 0.5): Reduce confidence by `decay_rate`
4. **Deprecate** (confidence < `deprecation_threshold`): Mark `deprecated=True`, collect task_types for redistillation candidates

### OutcomeTracker

Records task outcomes correlated with skill retrievals:

<details>
<summary>Code: OutcomeTracker</summary>

```python
class OutcomeTracker:
    def record_outcome(self, skill_id: str, task_id: str, success: bool): ...
    def get_skill_effectiveness(self, skill_id: str) -> float:
        # Returns success rate [0.0, 1.0], or 0.5 if no data
```

</details>

### EvolutionReport

Each cycle produces an `EvolutionReport`:

<details>
<summary>Code: EvolutionReport dataclass</summary>

```python
@dataclass
class EvolutionReport:
    skills_evaluated: int
    skills_promoted: int
    skills_decayed: int
    skills_deprecated: int
    redistillation_candidates: List[str]  # Task types needing new skills
```

</details>

### Population Health

`get_evolution_summary()` provides aggregate health metrics:

<details>
<summary>Data: evolution summary example</summary>

```python
{
    "total_active": 45,
    "avg_confidence": 0.72,
    "by_type": {
        "routing": {"count": 15, "avg_confidence": 0.78},
        "failure_lesson": {"count": 10, "avg_confidence": 0.65},
        ...
    }
}
```

</details>

</details>

## Replay Harness Integration

The replay evaluation harness now tests skill configurations alongside retrieval and scoring parameters. `SkillAwareReplayEngine` extends the standard `ReplayEngine` to track skill retrieval effectiveness during offline replay -- how many skills were retrieved per step, what types, and how much of the token budget they consumed. Old `DesignCandidate` entries without a `skill_config` field deserialize cleanly with `skill_config=None`.

<details>
<summary>Replay engine and skill metrics</summary>

### SkillBankConfig on DesignCandidate

The replay evaluation harness (see [Ch15](15-memrl-system.md)) now tests skill configurations alongside retrieval and scoring parameters:

<details>
<summary>Code: DesignCandidate with SkillBankConfig</summary>

```python
@dataclass
class DesignCandidate:
    retrieval_config: RetrievalConfig
    scoring_config: ScoringConfig
    staged_config: Optional[StagedConfig]
    skill_config: Optional[SkillBankConfig]  # NEW
```

</details>

### SkillBankConfig Parameters

<details>
<summary>Code: SkillBankConfig</summary>

```python
@dataclass
class SkillBankConfig:
    enabled: bool = True
    general_skills_max: int = 6
    task_specific_k: int = 6
    min_similarity: float = 0.4
    min_confidence: float = 0.3
    max_prompt_tokens: int = 1500
```

</details>

### SkillAwareReplayEngine

Extends `ReplayEngine` to evaluate skill retrieval effectiveness during offline replay:

1. Runs standard replay (chronological trajectory processing)
2. At each step, retrieves skills using `SkillRetriever` with the candidate's `SkillBankConfig`
3. Tracks skill coverage, types retrieved, and estimated context tokens
4. Produces `SkillReplayMetrics` alongside standard `ReplayMetrics`

<details>
<summary>Code: SkillReplayMetrics</summary>

```python
@dataclass
class SkillReplayMetrics:
    base_metrics: ReplayMetrics
    avg_skills_per_step: float        # Skills retrieved per trajectory
    total_skills_retrieved: int
    skill_coverage: float             # Fraction of steps with >=1 skill
    avg_skill_context_tokens: float   # Token budget consumption
    skills_by_type: Dict[str, int]    # Breakdown by skill type
```

</details>

### Backward Compatibility

Old `DesignCandidate` JSON without `skill_config` deserializes to `skill_config=None`, preserving forward compatibility with existing archive entries.

</details>

## Feature Flag

SkillBank is gated behind the `ORCHESTRATOR_SKILLBANK=1` environment variable and depends on the `memrl` feature flag for episodic store and embedder infrastructure. All skill retrieval paths are wrapped in try/except with debug logging, so if anything goes wrong, the system falls back gracefully to pre-SkillBank behavior with no skill context injected.

<details>
<summary>Feature flag and graceful degradation</summary>

```bash
ORCHESTRATOR_SKILLBANK=1  # Enable SkillBank retrieval + prompt injection
```

Dependency: requires `memrl` feature flag (SkillBank depends on episodic store and embedder infrastructure).

### Graceful Degradation

All skill retrieval paths are wrapped in try/except with debug logging. If SkillBank fails:
- Routing proceeds normally via `HybridRouter.route()`
- No skill context is injected into prompts
- The system operates identically to pre-SkillBank behavior

</details>

## Implementation

The SkillBank implementation spans roughly 2,020 lines of new code across 8 files, plus modifications to 10 existing files for integration. The end-to-end wiring covers CLI seeding, API data flow, anomaly detection, diagnostic records, debugger replay, and outcome tracking. Test coverage totals 139 tests, all running in-memory with no live inference required.

<details>
<summary>File inventory and integration points</summary>

### Files Created

| File | LOC | Purpose |
|------|-----|---------|
| `orchestration/repl_memory/skill_bank.py` | ~470 | Core SkillBank (SQLite + FAISS, CRUD, dedup, search) |
| `orchestration/repl_memory/skill_retriever.py` | ~180 | Two-level retrieval + prompt formatting |
| `orchestration/repl_memory/skill_evolution.py` | ~280 | EvolutionMonitor, OutcomeTracker, EvolutionReport |
| `orchestration/repl_memory/distillation/teachers.py` | ~280 | TeacherModel protocol, 3 teachers + MockTeacher |
| `orchestration/repl_memory/distillation/prompts.py` | ~130 | Success/failure/escalation prompt templates |
| `orchestration/repl_memory/distillation/pipeline.py` | ~250 | DistillationPipeline (batching, parsing, dedup) |
| `orchestration/repl_memory/distillation/failure_bridge.py` | ~200 | FailureBridge (Kuzu ↔ SkillBank) |
| `orchestration/repl_memory/replay/skill_replay.py` | ~230 | SkillAwareReplayEngine, SkillBankConfig, metrics |

### Files Modified

| File | Change |
|------|--------|
| `src/features.py` | Added `skillbank` feature flag |
| `src/api/state.py` | Added `skill_bank`, `skill_retriever` fields |
| `src/api/routes/chat_utils.py` | Added `skill_context` to RoutingResult |
| `src/api/routes/chat_pipeline/routing.py` | Skill-augmented routing branch |
| `src/api/routes/chat_pipeline/direct_stage.py` | Skill context prompt injection |
| `src/api/routes/chat_pipeline/repl_executor.py` | Skill context prompt injection |
| `src/api/services/memrl.py` | SkillBank lazy-loading and initialization |
| `orchestration/repl_memory/retriever.py` | Added SkillAugmentedRouter |
| `orchestration/repl_memory/faiss_store.py` | Custom filename parameters |
| `orchestration/repl_memory/replay/candidates.py` | Added SkillBankConfig to DesignCandidate |

### End-to-End Integration (2026-02-14)

The SkillBank infrastructure is wired into the seeding pipeline and ClaudeDebugger:

| Integration Point | File | How |
|-------------------|------|-----|
| CLI bootstrap | `scripts/skillbank/seed_skills.py` | `--teacher claude\|codex\|mock` → DistillationPipeline → SkillBank |
| API data flow | `RoutingResult.skill_ids` → `ChatResponse.skill_ids` → `RoleResult.skill_ids` | All 8 ChatResponse sites propagate |
| Anomaly detection | `anomaly.py` | +2 signals: `skill_mismatch` (0.5), `no_skills_available` (0.3) |
| Diagnostic records | `diagnostic.py` | `skill_retrieval` block with counts, types, token budget |
| Debugger replay | `claude_debugger.py` | `SkillAwareReplayEngine` first, fallback to `ReplayEngine` |
| Debugger display | `_build_prompt()` | Skills retrieved per diagnostic in Claude's context |
| Seeding replay | `seed_specialist_routing.py` | `--debug-replay` tries skill-aware replay, prints metrics |
| Outcome tracking | `OutcomeTracker` in seeding | `ORCHESTRATOR_SKILLBANK=1` records skill×task outcomes |
| Evolution | `--evolve` flag | Runs `EvolutionMonitor.run_evolution_cycle()` after seeding |

<details>
<summary>Data: test coverage breakdown</summary>

| Test File | Tests | Coverage |
|-----------|-------|----------|
| `tests/unit/test_skill_bank.py` | 37 | SkillBank CRUD, FAISS search, dedup, retrieval |
| `tests/unit/test_distillation.py` | 23 | Pipeline batching, teacher protocol, prompt building |
| `tests/unit/test_skill_integration.py` | 19 | Feature flag, router wrapping, state fields |
| `tests/unit/test_failure_bridge.py` | 15 | Failure context, mitigation sync, cross-reference |
| `tests/unit/test_skill_evolution.py` | 17 | Evolution cycles, outcome tracking, health metrics |
| `tests/unit/test_skill_replay.py` | 11 | Replay engine with skills, config serialization |
| `tests/unit/test_skill_diagnostics.py` | 17 | Anomaly signals, diagnostic builder integration |
| **Total** | **139** | All in-memory, no live inference |

</details>

</details>

## Performance Characteristics

Skill retrieval adds under 2ms of latency per request -- negligible compared to the existing 5-13ms episodic retrieval. The real win is token compression: structured skills use 30-80 tokens each versus 200-500 for raw trajectories, yielding a 5-15x compression ratio while allowing you to inject more memories per query.

<details>
<summary>Latency and token compression data</summary>

### Retrieval Latency (Estimated)

| Operation | Time | Notes |
|-----------|------|-------|
| General skills query | <1ms | SQLite indexed query |
| FAISS skill search (500 skills) | <0.5ms | O(log n), IndexFlatIP |
| Format for prompt | <1ms | String concatenation |
| **Total skill retrieval** | **<2ms** | Negligible vs existing 5-13ms retrieval |

### Token Compression

| Source | Tokens per Memory | Memories per Query |
|--------|-------------------|--------------------|
| Raw trajectory (current) | 200-500 | 5 |
| Structured skill (SkillBank) | 30-80 | 6-12 |
| **Compression ratio** | **5-15x** | More skills, fewer tokens |

</details>

## References

<details>
<summary>Literature and implementation references</summary>

### Primary

1. **[SkillRL]** Xia, P., Chen, J., Wang, H., Liu, J., Zeng, K., Wang, Y., Han, S., Zhou, Y., Zhao, X., Chen, H., Zheng, Z., Xie, C., & Yao, H. (2026). *SkillRL: Evolving Agents via Recursive Skill-Augmented Reinforcement Learning.* arXiv:2602.08234. https://arxiv.org/abs/2602.08234

2. **[ALMA]** Xiong, W. et al. (2026). *ALMA: Adaptive Learning for Memory Architectures.* (Motivates offline replay evaluation; referenced in [Ch15](15-memrl-system.md) replay harness.)

### Related Work (Context)

3. **[GRPO]** Shao, Z. et al. (2024). *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models.* Group Relative Policy Optimization — the RL algorithm used by SkillRL. Our system uses TD-learning (Q-value updates) rather than policy gradient RL, so we adapt the skill distillation mechanism without the RL training loop.

4. **[xRouter]** Qian, C. et al. (2025). *xRouter: Training Cost-Aware LLMs Orchestration System via Reinforcement Learning.* arXiv:2510.08439. (Referenced in [Ch25](25-cost-aware-rewards.md) QScorer cost-penalty design. SkillBank's escalation skills complement xRouter-style cost optimization.)

5. **[SimpleMem]** Simple memory baseline in SkillRL. Stores raw trajectories and retrieves by similarity. Equivalent to our pre-SkillBank EpisodicStore + TwoPhaseRetriever. SkillBank outperforms SimpleMem+GRPO by +25.8% on WebShop.

6. **[EvolveR]** Competitive baseline in SkillRL. Iterative self-refinement of agent behaviors. SkillRL outperforms by +4% on Search QA through structured abstraction rather than raw trajectory refinement.

### Implementation

7. `orchestration/repl_memory/skill_bank.py`: SkillBank core
8. `orchestration/repl_memory/skill_retriever.py`: Two-level retrieval
9. `orchestration/repl_memory/skill_evolution.py`: Recursive evolution
10. `orchestration/repl_memory/distillation/`: Distillation pipeline (teachers, prompts, pipeline, failure bridge)
11. `orchestration/repl_memory/replay/skill_replay.py`: Replay harness integration

</details>

---

*Previous: [Chapter 26: Claude-in-the-Loop Debugger](26-claude-debugger.md)* | *Next: (none — latest chapter)*
