# Chapter 17: Memory Seeding & Bootstrap

## Introduction

The MemRL system requires bootstrap data to function effectively. Without seed memories, the episodic store is empty and retrieval returns nothing. This chapter covers the seeding infrastructure that provides canonical examples, diverse exploration patterns, and graph-backed failure/hypothesis knowledge.

**Seeding philosophy:** Provide high-quality, high-Q-value canonical examples that enable immediate retrieval on common tasks, while also seeding diverse exploration patterns to prevent overfitting to simple cases.

## Seed Loader Architecture

The seed loader is the main entry point for populating the episodic store. It reads canonical examples from JSON, generates embeddings via BGE-large, and writes them into memory with high Q-values so the system can retrieve useful patterns from day one.

<details><summary>Core components and file locations</summary>

| Component | Purpose | Location |
|-----------|---------|----------|
| `seed_loader.py` | Main seeding script | `orchestration/repl_memory/seed_loader.py` |
| `seed_examples.json` | 56 canonical REPL examples | `orchestration/repl_memory/seed_examples.json` |
| `graph_seeds.yaml` | Failure modes & hypotheses | `orchestration/repl_memory/graph_seeds.yaml` |
| Seeding scripts | Diverse seeding strategies | `scripts/seed_*.py` (9 scripts) |

Each canonical example has the following structure:

<details><summary>Data: seed example JSON format</summary>

```json
{
  "task": "List files in a directory",
  "code": "result = list_dir('/path/to/dir')\nFINAL(result)",
  "tools_used": ["list_dir", "FINAL"],
  "category": "filesystem"
}
```

</details>

Categories include:
- `filesystem` - Directory listing, file info, peeking
- `search` - Grep patterns, function definitions, TODO comments
- `document` - OCR, PDF extraction, figure extraction
- `analysis` - Data parsing, log analysis
- `computation` - Math, statistics, aggregations

</details>

<details><summary>Loading canonical seeds into memory</summary>

**What it does:**

1. Loads 56 examples from `seed_examples.json`
2. Generates embeddings using `TaskEmbedder` (BGE-large, 1024-dim)
3. Stores in episodic memory with:
   - `action`: The code snippet
   - `action_type`: "exploration"
   - `outcome`: "success"
   - `initial_q`: 0.9 (high Q-value for canonical examples)
   - `context`: `{"is_seed": True, "category": "...", "tools_used": [...]}`

<details><summary>Code: basic usage and expected output</summary>

```bash
# First-time seeding
python orchestration/repl_memory/seed_loader.py

# Force reload (clears existing memories)
python orchestration/repl_memory/seed_loader.py --force
```

**Output:**

```
Loading 56 seed examples...
  Loaded 10/56 examples...
  Loaded 20/56 examples...
  ...
Seeding complete!
  Loaded: 56
  Failed: 0
  By category: {'filesystem': 20, 'search': 12, 'document': 15, 'analysis': 6, 'computation': 3}

Memory stats:
  Total memories: 56
  FAISS embeddings: 56
  Average Q-value: 0.90
```

</details>

</details>

## Seeding Strategies

The system provides 9 specialized seeding scripts, each targeting a different aspect of the memory distribution. Together they build a balanced store of successes, failures, complex chains, and probabilistic explorations so the agent does not overfit to simple canonical patterns.

<details><summary>Strategy 1: Diverse memories</summary>

**Script:** `seed_diverse_memories.py`

Prevents overfitting to simple cases by seeding complex multi-step tasks. Generates 100-500 memories with increasing complexity, combining multiple tools in sequence, and including conditional logic and error handling. Q-values range from 0.7-0.9 (high but not canonical).

<details><summary>Code: multi-step example</summary>

```python
# Complex multi-step: OCR -> grep -> analysis
doc = json.loads(ocr_document('/path/to/paper.pdf'))
matches = grep('algorithm', file_path='/tmp/extracted.txt')
result = analyze_pattern(matches)
FINAL(result)
```

</details>

</details>

<details><summary>Strategy 2: Failure memories</summary>

**Script:** `seed_failure_memories.py`

Seeds common failure patterns with Q-values below 0.5 to teach the system what NOT to do. Generates 50-100 failed attempts covering common mistakes like wrong tool usage, missing imports, and path errors. Outcome is "failure" and Q-values are 0.1-0.4 to discourage repetition.

<details><summary>Code: example failure pattern</summary>

```python
# BAD: Uses Python imports instead of REPL tools
import os
files = os.listdir('/tmp')  # Will trigger security error
FINAL(files)
```

</details>

</details>

<details><summary>Strategy 3: Diverse failures</summary>

**Script:** `seed_diverse_failures.py`

Seeds complex failure chains and recovery patterns. Generates multi-step failures (A fails, B attempted, B also fails), includes partial successes (step 1 works, step 2 fails), and links failures to the graph via `failure_graph.record_failure()`.

</details>

<details><summary>Strategy 4: Probabilistic memories</summary>

**Script:** `seed_probabilistic_memories.py`

Seeds exploration with randomized Q-values to model uncertainty. Uses the same tasks as canonical examples, but Q-values are drawn from a Beta(2, 5) distribution (mean ~0.3, with variance) and outcomes are randomized: 70% success, 30% failure. This prevents overconfidence in untested scenarios.

</details>

<details><summary>Strategy 5: Decomposition memories</summary>

**Script:** `seed_decomposition_memories.py`

Seeds task decomposition patterns that map high-level goals to sub-tasks. Generates 50 examples of task hierarchies including planning, subtask execution, and aggregation. Q-values are 0.8 because decomposition is especially valuable.

<details><summary>Code: decomposition example</summary>

```python
# High-level: Analyze all PDFs in directory
files = list_dir('/docs/')
results = []
for f in filter(lambda x: x.endswith('.pdf'), files):
    doc = ocr_document(f)
    results.append(summarize(doc))
FINAL(aggregate(results))
```

</details>

</details>

<details><summary>Strategy 6: Memory from logs</summary>

**Script:** `seed_memory_from_logs.py`

Bootstraps memory from real agent activity logs. Parses `logs/agent_audit.log`, extracts task-action pairs from successful executions, assigns Q-values based on outcome (1.0 for success, 0.2 for failure), and filters out low-quality entries like truncated or error messages.

<details><summary>Code: log-based seeding usage</summary>

```bash
python scripts/seed_memory_from_logs.py --log-file logs/agent_audit.log --min-quality 0.5
```

</details>

</details>

<details><summary>Strategy 7: Success patterns</summary>

**Script:** `seed_success_patterns.py`

Seeds known-good patterns from benchmark results. Extracts successful action sequences from benchmark JSON, focusing on high-scoring runs (Claude-as-Judge score of 3 or above), and assigns Q-values of 0.8-1.0 based on the benchmark score.

</details>

<details><summary>Strategy 8: Graph seeds</summary>

**Script:** `seed_graphs.py`

Loads failure modes and hypotheses into the graph databases. Parses `graph_seeds.yaml`, creates FailureMode, Symptom, and Mitigation nodes, creates Hypothesis nodes with initial confidence, and links them to episodic memory where applicable.

<details><summary>Code: graph seeding usage and output</summary>

```bash
python scripts/seed_graphs.py --force
```

**Output:**

```
Loading failure modes...
  Created 14 failure modes
  Created 45 symptom patterns
  Created 16 mitigations

Loading hypotheses...
  Created 15 hypotheses
  Average initial confidence: 0.78

Graph stats:
  Failure graph: 14MB (75 nodes, 120 edges)
  Hypothesis graph: 4.6MB (15 nodes, 30 edges)
```

</details>

</details>

<details><summary>Strategy 9: Remaining Phase B</summary>

**Script:** `seed_remaining_phase_b.py`

Seeds incomplete Phase B implementation tasks (specialist workflows). Generates placeholder memories for unimplemented features with Q-values of 0.3 (uncertain, needs validation) and marks them with `{"phase": "B", "status": "pending"}`.

</details>

<details><summary>Strategy 10: 3-Way routing evaluation</summary>

**Script:** `seed_specialist_routing.py --3way`

Trains the frontdoor for faithful probability estimation via 3-way comparative testing. Each question runs through 4 configurations: `SELF:direct` (frontdoor, no tools), `SELF:repl` (frontdoor/vision worker with tools, delegation disabled), `ARCHITECT` (dual-architect best-of-two evaluation), and `WORKER` (scored indirectly via delegation chains). Binary rewards (1.0 pass, 0.0 fail) are used instead of cost-weighted rewards so the system learns true P(success). Cost metrics are stored in metadata for later Optuna optimization.

Infrastructure errors (timeouts, connection failures) produce no reward -- the action is skipped and retried next batch. For VL questions, `SELF:repl` is `worker_vision:repl` (legacy `worker_vision:react` is backward-compatible in historical reward parsing).

<details><summary>Code: 3-way seeding commands</summary>

```bash
# Full 3-way seeding run
python scripts/benchmark/seed_specialist_routing.py --3way --suites thinking coder --sample-size 20

# Dry run (no reward injection)
python scripts/benchmark/seed_specialist_routing.py --3way --dry-run --suites thinking --sample-size 5
```

</details>

**Key difference from comparative seeding:**
- Comparative seeding uses cost-weighted rewards
- 3-way seeding uses binary rewards for faithful P(success) estimation
- Cost is stored in metadata, not incorporated into Q-values

</details>

<details><summary>Question pool and pre-extracted data</summary>

All ~53K questions from 18 HF dataset adapters plus YAML suites are pre-extracted into `benchmarks/prompts/question_pool.jsonl`. Runtime sampling reads this file (~100ms) instead of loading 16 Arrow/Parquet datasets (~30s).

- **Sampling**: Full shuffle per suite, take first N unseen. Guarantees coverage of entire pool.
- **Seen tracking**: `benchmarks/results/eval/seen_questions.jsonl` -- questions marked seen only when rewards are injected.
- **Debug mode** (`--debug`): When a suite is exhausted, backfills with seen questions (via `allow_reseen`). Normal mode skips exhausted suites.
- **Reset**: `scripts/session/reset_episodic_memory.sh` clears episodic DB + FAISS + seen set.
- **Rebuild**: `--rebuild-pool` re-extracts from all adapters.

</details>

<details><summary>Claude-in-the-Loop debugger</summary>

The `--debug` flag (requires `--3way`) enables automatic pipeline debugging via a persistent Claude Code session. See [Chapter 26](26-claude-debugger.md) for full documentation: 17 anomaly signals, hot-swap/code fixes, 3-phase regression suite (verify/generalize/regress), MemRL interaction (TD-learning on retried questions), auto-discovery of new failure patterns, and audit trail.

**Anomaly signals (17):** repetition_loop, comment_only, template_echo, self_doubt_loop, format_violation, think_tag_leak, near_empty, excessive_tokens, delegation_format_error, self_escalation, vision_blindness, silent_execution, repl_no_tools, slow_delegation, function_repr_leak, status_phrase_final, misrouted_to_coder.

**Auto-discovery:** The debugger instructs Claude to propose new anomaly detectors via structured `NEW_SIGNAL:` output. Proposals are persisted to `logs/proposed_signals.jsonl` for human review and optional inclusion in `anomaly.py`.

**Retry persistence:** Retry queue survives script crashes via JSONL persistence (`logs/retry_queue.jsonl`). Previous sessions' pending retries are loaded on startup.

<details><summary>Code: debugger invocation commands</summary>

```bash
# Live debugging (Claude analyzes every 5 answers)
python scripts/benchmark/seed_specialist_routing.py --3way --continuous --debug

# With auto-commit of debugger fixes
python scripts/benchmark/seed_specialist_routing.py --3way --continuous --debug --debug-auto-commit

# Dry run (log diagnostics without invoking Claude)
python scripts/benchmark/seed_specialist_routing.py --3way --debug --debug-dry-run
```

</details>

</details>

## Seeding Order & Dependencies

Running the seeding scripts in the right order matters. Canonical examples go first to establish the high-confidence baseline, then graphs for failure knowledge, then progressively noisier data. Here is the recommended sequence.

<details><summary>Step-by-step seeding commands</summary>

1. **Canonical examples first** - High-quality baseline
   ```bash
   python orchestration/repl_memory/seed_loader.py --force
   ```

2. **Graph seeds** - Failure/hypothesis knowledge
   ```bash
   python scripts/seed_graphs.py --force
   ```

3. **Diverse patterns** - Prevent overfitting
   ```bash
   python scripts/seed_diverse_memories.py --count 200
   ```

4. **Failure patterns** - Learn what NOT to do
   ```bash
   python scripts/seed_failure_memories.py --count 50
   python scripts/seed_diverse_failures.py --count 50
   ```

5. **Real logs** - Bootstrap from production
   ```bash
   python scripts/seed_memory_from_logs.py --min-quality 0.6
   ```

6. **Success patterns** - Benchmark-driven
   ```bash
   python scripts/seed_success_patterns.py --min-score 3
   ```

</details>

## Memory Distribution After Seeding

After running the full seeding pipeline, you should see roughly 500-1000 memories with a balanced mix of high-confidence patterns, exploratory variety, and anti-patterns. The average Q-value should land around 0.65.

<details><summary>Expected distribution by source</summary>

| Source | Count | Avg Q-value | Purpose |
|--------|-------|-------------|---------|
| Canonical examples | 56 | 0.90 | High-confidence patterns |
| Diverse memories | 200 | 0.75 | Exploration variety |
| Failure memories | 100 | 0.25 | Anti-patterns |
| Log-based | 50-500 | 0.60 | Real usage patterns |
| Benchmark-driven | 100-200 | 0.85 | Proven solutions |
| **Total** | **500-1000** | **0.65** | Balanced coverage |

</details>

## Verification

You can check seeding status at any time by querying the episodic store directly. If the total count and average Q-value look reasonable, seeding is healthy.

<details><summary>Code: checking seeding status</summary>

```python
from orchestration.repl_memory.episodic_store import EpisodicStore

store = EpisodicStore()
stats = store.get_stats()

print(f"Total memories: {stats['total_memories']}")
print(f"Average Q-value: {stats['overall_avg_q']:.2f}")
print(f"Recent successes: {stats.get('recent_success_rate', 0):.0%}")
```

**Expected output:**

```
Total memories: 856
Average Q-value: 0.67
Recent successes: 78%
```

</details>

## 3-Way Action Keys (February 2026)

The 3-way evaluation mode uses a distinct action vocabulary that maps to routing decisions. These keys are stored in episodic memory and the HybridRouter's `route_3way()` method retrieves memories by them.

<details><summary>Action key definitions</summary>

| Action Key | What It Represents | Source Role | Mode |
|------------|-------------------|-------------|------|
| `SELF:direct` | Frontdoor without tools | frontdoor | direct |
| `SELF:repl` | Frontdoor with tools | frontdoor | repl |
| `ARCHITECT` | Architect with delegation | architect_general + architect_coding (best-of-two) | delegated |
| `WORKER` | Worker models | via delegation | -- |

</details>

## Infra Safeguards (2026-02-07)

Recent seeding regressions showed that a single stalled heavy-model request can block the orchestrator event loop and cascade into 600s timeouts. The infrastructure now guards against this with a CPU-exclusive inference lock (heavy models acquire exclusive; workers/embedders acquire shared and only run when no heavy model is active), async safety (all blocking LLM calls offloaded from the event loop), 3-way timeout cleanup (slot erasure on infra timeouts to prevent stuck backends), and backend probes in `/health` to detect hung backends even when circuit state is stale.

## Architect Delegation in 3-Way Eval (2026-02-09)

The 3-way ARCHITECT evaluation runs `architect_general` and `architect_coding` in delegated mode, where the architect decides via TOON whether to answer directly (`D|answer`) or delegate to a specialist (`I|brief:<spec>|to:coder_escalation`).

<details><summary>Known issue, fix, and slot-erase details</summary>

**Known issue (fixed):** The original architect prompt presented `D|` and `I|` as side-by-side template examples. Qwen3-235B echoed both, causing `_extract_toon_decision` to find `D|Answer` first and parse it as a direct answer. The delegation chain to `coder_escalation` (port 8081, Qwen2.5-Coder-32B) was never exercised.

**Fix:** Prompt restructured as bullet-list alternatives with "EXACTLY ONE line" guard. Architect now correctly delegates code tasks and provides architectural design briefs (approach, data structures, algorithm, complexity) for the coding specialist.

**Slot-erase for stuck backends:** When the seeding script's HTTP client times out, the llama-server may still be generating. `_erase_slots(port)` sends `POST /slots/{id}?action=erase` to cancel in-progress inference. If the server is stuck in prompt eval, the erase request itself may hang. The `_SLOT_ERASE_CAPABILITY` cache tracks which ports support slot erasure and disables erase attempts on ports that return 404/405/501.

</details>

## Timeout + Telemetry Updates (2026-02-08)

Several improvements landed to handle timeout cascades and improve observability during 3-way evaluation runs.

<details><summary>Timeout and telemetry implementation details</summary>

- **Adaptive per-call timeout budget** in 3-way eval: timeout is selected by role/mode/modality and capped by CLI `--timeout` (hard ceiling). This reduces worst-case stall wait while preserving headroom for slow architect paths.
- **Observed-runtime timeout bumping**: REPL and architect calls can be raised using earlier per-question observed latency (direct/repl), reducing false `INFRA` on hard long-generation tasks while still respecting the hard ceiling.
- **Structured error normalization**: benchmark caller now maps `error_code`/`error_detail` into a unified `error` field.
- **Telemetry consistency invariant**: `tools_used`, `tools_called`, and `tool_timings` are normalized and kept internally consistent for debugging and post-hoc analysis.
- **Slot-erase capability guard**: 3-way cleanup now detects unsupported `/slots/{id}?action=erase` behavior on llama-server builds and disables repeated failing erase attempts instead of logging false success.
- **Live slot progress polling (2026-02-09)**: forced 3-way calls poll backend `/slots` during execution and emit `[slot-progress]` logs with task id + decoded token counters.
- **INFRA token estimate (2026-02-09)**: when API returns `0 tok` under timeout/disconnect, seeding records `tokens_generated_estimate` from slot counters and surfaces it in logs (`0 tok, est N tok`).

</details>

## Skill Seeding (February 2026)

The SkillBank extends the seeding philosophy to structured skills. Initial SkillBank bootstrap distills existing high-Q episodic memories into compressed, reusable skills via the DistillationPipeline. Think of it as graduating raw memories into polished, retrievable skill definitions.

<details><summary>Relationship to canonical seeds and bootstrap process</summary>

### Relationship to Canonical Seeds

Canonical seed examples (56 REPL examples at Q=0.9) are the **raw material** for initial skill distillation. The distillation pipeline processes these high-confidence trajectories first, producing the foundational skill set that covers common task patterns.

### Bootstrap Process

1. Extract high-Q trajectories from episodic store (`Q >= 0.7`)
2. Batch through DistillationPipeline with selected teacher model
3. Deduplicate against existing skills (cosine similarity > 0.85)
4. Store in SkillBank with initial confidence from teacher analysis

This is analogous to SkillRL's SFT cold-start -- bootstrapping structured knowledge from existing experience without model weight updates.

See [Chapter 27](27-skillbank-experience-distillation.md) for full SkillBank documentation.

</details>

<details><summary>References</summary>

- **Seed loader**: `orchestration/repl_memory/seed_loader.py`
- **Canonical examples**: `orchestration/repl_memory/seed_examples.json`
- **Graph seeds**: `orchestration/repl_memory/graph_seeds.yaml`
- **Seeding scripts**: `scripts/seed_*.py` (9 scripts)
- **3-way seeding**: `scripts/benchmark/seed_specialist_routing.py --3way`
- **Seeding types**: `scripts/benchmark/seeding_types.py` (action keys, cost tiers)
- **Seeding rewards**: `scripts/benchmark/seeding_rewards.py` (binary rewards)
- **EpisodicStore**: `orchestration/repl_memory/episodic_store.py`

</details>

---

*Previous: [Chapter 16: Graph-Based Reasoning](16-graph-reasoning.md)* | *Next: [Chapter 18: Escalation & Routing](18-escalation-and-routing.md)*
