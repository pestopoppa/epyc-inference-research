# Chapter 26: Claude-in-the-Loop Debugger

## Introduction

Automatic pipeline debugging during 3-way evaluation. A persistent Claude Code session monitors inference diagnostics, identifies anomalies, applies prompt/code fixes via hot-swap, and verifies fixes with a 3-part mini regression suite.

## Architecture Overview

The debugger runs as a background subprocess inside the eval loop. When `seed_specialist_routing.py` evaluates a question, it builds a diagnostic record, batches it, and hands it off to a persistent Claude session that can edit prompts or code in real time. After any fix, a mini regression suite re-tests the failed question plus fresh and previously-passing questions to make sure nothing broke.

<details>
<summary>Component flow diagram</summary>

```
seed_specialist_routing.py (eval loop)
  ├── evaluate_question_3way()  → role_results, rewards, metadata
  ├── build_diagnostic()        → anomaly signals + raw answer + tap offset
  ├── debugger.add_diagnostic() → batches diagnostics (default: 5 per batch)
  ├── debugger.end_question()   → flushes urgent batches early
  │
  │   [Background: Claude subprocess]
  │   ├── snapshot files (SHA-1 of first 8KB)
  │   ├── Claude analyzes diagnostics + tap output
  │   ├── Claude edits .md prompts or .py code
  │   ├── diff snapshots → detect actual changes
  │   ├── hot-restart API if .py changed
  │   ├── queue failed questions for retry
  │   └── record to change_log (JSONL audit trail)
  │
  └── pop_retries() → mini regression suite
      ├── VERIFY:     exact failed questions
      ├── GENERALIZE: 2 fresh unseen per affected suite
      └── REGRESS:    2 previously-passing per affected suite
```

</details>

## 12 Anomaly Signals

Each answer gets scored by 12 detectors in `src/pipeline_monitor/anomaly.py`. Signals with weight 1.0 are considered urgent and trigger an early flush to Claude. The final anomaly score is the max of all triggered weights, so a single urgent signal immediately escalates the batch.

<details>
<summary>Signal definitions and weights</summary>

| Signal | Weight | Trigger |
|--------|--------|---------|
| `repetition_loop` | 1.0 | Trigram unique ratio < 0.4 (degeneration) |
| `template_echo` | 1.0 | Both `D\|` AND `I\|` prefixes in same answer |
| `format_violation` | 1.0 | Architect role, no `D\|` or `I\|`, answer > 50 chars |
| `near_empty` | 1.0 | Answer < 5 tokens, no error (except MCQ) |
| `delegation_format_error` | 1.0 | `I\|` present but missing `brief:` field |
| `vision_blindness` | 1.0 | Vision role, < 10 tokens |
| `comment_only` | 0.5 | All code lines are `#`-prefixed or blank |
| `self_doubt_loop` | 0.5 | > 3 restart phrases ("Actually", "Wait", "Let me reconsider"...) |
| `think_tag_leak` | 0.5 | `<think>` in answer text |
| `excessive_tokens` | 0.5 | MCQ with > 2000 tokens generated |
| `self_escalation` | 0.5 | Consecutive duplicate roles in role_history |
| `silent_execution` | 0.5 | Tools used >= 1, no error, empty answer |

**Scoring**: `anomaly_score = max(triggered weights)`. Range [0, 1]. Score >= 1.0 sets urgent flag.

</details>

## Batching & Background Execution

The debugger accumulates diagnostics and dispatches them to Claude in batches so the eval loop does not block. Diagnostics pile up until the batch is full (default 5), then Claude gets them all at once. If any signal has weight 1.0, the batch flushes early -- you do not want to wait around when the pipeline is producing garbage output.

<details>
<summary>Dispatch lifecycle details</summary>

1. `add_diagnostic(diag)` appends to batch, collects any finished background result
2. When `len(batch) >= batch_size` (default 5), calls `_dispatch()`:
   - Captures file snapshot (SHA-1 hashes of dirty files)
   - Launches `claude --session-id {id} --json` as background subprocess
   - First invocation includes full system prompt; subsequent reuse session for context accumulation
3. `end_question()` triggers early flush if `_urgent=True` (any signal with weight 1.0)
4. Background Claude runs concurrently -- eval loop continues scoring the next question

</details>

### Prompt Construction

Each batch prompt gives Claude everything it needs to diagnose the failure: the question metadata, the anomaly signals, and crucially the raw inference and REPL logs so it can see exactly what the model said and what the code executed. Both logs are captured via byte-range offsets recorded around each `/chat` API call in `seeding_eval.py`, then read inline by `_read_tap_inline()` in the debugger.

<details>
<summary>Per-diagnostic fields included in the prompt</summary>

Each batch prompt includes per-diagnostic:
- Question ID, pass/fail, config, role, mode
- Expected answer, scoring method, tokens generated, error
- Triggered anomalies + overall score
- Full role history and tool calls
- **Tool chain diagnostics** (when present): chain IDs, requested/used mode, wave count, fallback status, and parallel-mutation flag
- **Inference log** (inlined, up to 12K chars): raw prompt/response for every LLM call in the delegation chain -- architect TOON decisions, specialist outputs, escalation triggers
- **REPL execution log** (inlined, up to 4K chars): code executed and stdout/stderr -- NameErrors, SyntaxErrors, import failures, silent execution issues
- Full answer text (truncated at 2000 chars)

If a tap file is missing or offset is stale (e.g., TUI restart), the debugger gracefully omits the log section.

</details>

## Fix Application & Hot-Restart

Once Claude's subprocess finishes, the debugger diffs file snapshots to see what actually changed, then applies the appropriate restart strategy. Prompt edits are picked up automatically on the next inference call (they are just markdown files read from disk). Code edits require a uvicorn restart, which takes about 10 seconds.

<details>
<summary>Fix application steps</summary>

1. **Snapshot diff**: Compare file hashes before/after -- detects what Claude actually changed vs pre-existing dirty state
2. **Prompt hot-swap**: If `.md` files changed in `orchestration/prompts/`, next inference request picks up the edit automatically (no restart needed, ~1ms disk read)
3. **Code hot-restart**: If `.py` files changed, runs `orchestrator_stack.py reload orchestrator` (uvicorn restart, ~10s)
4. **Auto-commit** (optional, `--debug-auto-commit`): `git add -A && git commit` per batch, enabling easy `git revert`

</details>

## Retry-After-Fix: Mini Regression Suite

After Claude applies fixes that modify files, the eval loop runs a 3-part verification to prevent overfitting. This is the key mechanism that keeps the debugger honest -- it cannot just tailor a fix to the one question that failed.

<details>
<summary>Three-phase verification process</summary>

### Phase 1: VERIFY
Re-test the exact questions that failed and triggered the fix. Confirms the fix addresses the original failure. Each question retried at most once (tracked by `_retried` set).

### Phase 2: GENERALIZE
Sample 2 fresh unseen questions per affected suite. These questions were never seen by Claude -- it cannot tailor fixes to them. If verify passes but generalize fails, the fix was too narrow.

### Phase 3: REGRESS
Sample up to 2 previously-passing questions per affected suite. If these now fail, the fix introduced a regression.

### Metadata

All retry results carry:

<details>
<summary>Code: retry metadata fields</summary>

```python
meta["is_retry"] = True
meta["retry_tag"] = "verify" | "generalize" | "regress"
meta["retry_batch_id"] = N  # which batch triggered the fix
```

</details>

Retry diagnostics feed back into the debugger. Generalize/regress failures can trigger new fix cycles. Verify questions are capped at 1 retry to prevent infinite loops.

</details>

## MemRL Interaction

Debug mode injects rewards into episodic memory (`--debug` is orthogonal to `--dry-run`). This means debug runs simultaneously seed MemRL while fixing bugs. The interesting part is how Q-values converge when the same question gets retried after a fix -- the TD-learning update naturally produces a lower Q-value for questions that needed a pipeline fix, which is exactly the right signal.

<details>
<summary>Q-value convergence mechanics</summary>

### Q-Value Convergence on Retried Questions

When a question is retried after a fix, the same `(task_description, action)` pair gets a second reward injection:

1. **Pre-fix** (failed): `_inject_3way_rewards_http(reward=0.0)` -> `score_external_result()` creates memory with `initial_q = 0.5`
2. **Post-fix** (passes): `_inject_3way_rewards_http(reward=1.0)` -> `score_external_result()` finds existing memory (cosine similarity >= 0.85, same action) -> TD update

The TD-learning update:

<details>
<summary>Code: TD update formula</summary>

```
Q_decayed = 0.5 + (Q_old - 0.5) * decay_rate ^ days_elapsed
Q_new = Q_decayed + α(reward - Q_decayed)
```

</details>

With alpha=0.1 and both updates in the same run (days_elapsed ~ 0):
- After fail:  Q = 0.50
- After pass:  Q = 0.50 + 0.1(1.0 - 0.50) = 0.55

This is **intentional**: a question that required a pipeline fix to pass should have a lower Q-value than one that passed on first attempt (Q ~ 1.0). MemRL correctly learns "this task type is harder for this route."

### Generalize Questions

Generalize questions are fresh -- they create new memories entirely. If they pass post-fix, their Q-values start at `0.5 + 1.0 * 0.5 = 1.0` -- clean signals uncontaminated by pre-fix failures.

</details>

## Audit Trail

All debugger activity is logged to two JSONL files: one for fix decisions and reasoning, one for per-answer diagnostics. The fix log is file-locked (POSIX `fcntl.flock`) and append-only, with full git diffs stored so you can rewind any change with `git apply --reverse`.

<details>
<summary>Log format and locations</summary>

Fix log at `/mnt/raid0/llm/claude/logs/debug_changes.jsonl`:

<details>
<summary>Data: change log JSONL schema</summary>

```json
{
  "ts": "2026-02-10T03:14:15.926535",
  "session_id": "abc123",
  "batch_id": 7,
  "questions_analyzed": ["math/q42", "coder/q17"],
  "anomalies_seen": {"math/q42": {"template_echo": true}},
  "claude_reasoning": "Template echo: architect outputs both D| and I| ...",
  "files_modified": ["orchestration/prompts/architect_investigate.md"],
  "git_diff": "diff --git a/...",
  "git_commit_sha": "a1b2c3d"
}
```

</details>

Diagnostics per-answer are logged separately to `/mnt/raid0/llm/claude/logs/seeding_diagnostics.jsonl`.

Reload watchdog events are emitted to normal logs by `ClaudeDebugger`:

```bash
# Triage debugger-triggered service reloads
rg -n "\\[DEBUG\\]\\[RELOAD\\] (START|OK|FAIL|UNHEALTHY|EXCEPTION)" logs/*.log
```

</details>

## CLI Flags

You can control debug behavior entirely through command-line flags. The most common mode is `--debug` by itself, which enables live analysis with default settings. Add `--debug-dry-run` when you want diagnostics logged without Claude actually touching anything.

<details>
<summary>Available flags and examples</summary>

<details>
<summary>Code: CLI usage examples</summary>

```bash
# Live debugging (Claude analyzes every 5 answers, applies fixes)
python scripts/benchmark/seed_specialist_routing.py --3way --continuous --debug

# Dry run (log diagnostics, don't invoke Claude)
python scripts/benchmark/seed_specialist_routing.py --3way --debug --debug-dry-run

# Auto-commit each fix batch
python scripts/benchmark/seed_specialist_routing.py --3way --debug --debug-auto-commit

# Add MemRL replay context to debugger prompts (+ periodic replay metrics in continuous mode)
python scripts/benchmark/seed_specialist_routing.py --3way --debug --debug-replay

# Custom batch size and threshold
python scripts/benchmark/seed_specialist_routing.py --3way --debug --debug-batch-size 10 --debug-threshold 0.5
```

</details>

</details>

## Extended Observation Patterns (February 2026)

Diagnostic records now include additional tunable fields that give Claude richer context about the orchestrator's decision-making. The prompt builder only includes these when they carry non-default values, so simple cases stay compact. The most useful pattern to watch for: `think_harder_attempted=True, think_harder_succeeded=False` tells you the pipeline already tried its escalation strategy and still failed -- Claude should look for prompt or tool issues rather than suggesting "try harder."

<details>
<summary>Extended diagnostic fields</summary>

| Field | Type | Description |
|-------|------|-------------|
| `cost_dimensions` | dict | Breakdown of cost signals: model tier, memory tier, elapsed ratio, tokens |
| `think_harder_attempted` | bool | Whether the pipeline invoked extended thinking mode |
| `think_harder_succeeded` | bool | Whether extended thinking produced a correct answer after standard mode failed |
| `cheap_first_attempted` | bool | Whether cheap-first cascade was tried (frontdoor before escalation) |
| `cheap_first_passed` | bool | Whether cheap-first produced a correct answer (no escalation needed) |
| `grammar_enforced` | bool | Whether json_schema/grammar constraint was active on the request |
| `parallel_tools_used` | int | Number of parallel tool invocations in REPL mode |
| `cache_affinity_bonus` | float | RadixAttention cache hit benefit (0.0 = miss, 1.0 = full prefix cached) |

The ClaudeDebugger prompt builder (`_build_batch_prompt()`) conditionally includes these fields only when they carry non-default values. This keeps batch prompts compact for simple cases while giving Claude full visibility into complex routing and cost decisions when debugging failures.

</details>

## Skill Diagnostics (February 2026)

When SkillBank is enabled (`ORCHESTRATOR_SKILLBANK=1`), the debugger can surface skill health information alongside anomaly signals. This is a future integration point -- the wiring exists, but the diagnostic enrichment and recommendation logic are not yet fully automated.

<details>
<summary>Skill diagnostic enrichment and recommendations</summary>

### Diagnostic Enrichment

The `build_diagnostic()` function can include skill retrieval data when available:
- **Skills retrieved**: Count and types of skills injected into the failing prompt
- **Skill confidence**: Average confidence of retrieved skills
- **Skill coverage**: Whether relevant skills existed for this task type

### Debugger Recommendations

Claude can recommend skill-related actions based on diagnostic patterns:
- **Low confidence skills retrieved**: Suggest evolution cycle or redistillation
- **High retrieval + low effectiveness**: Skill principle may be misleading -- flag for review
- **No skills for task type**: Gap in SkillBank coverage -- recommend distillation run

See [Chapter 27](27-skillbank-experience-distillation.md) for full SkillBank documentation.

</details>

## File Locations

<details>
<summary>References</summary>

| File | Purpose |
|------|---------|
| `src/pipeline_monitor/claude_debugger.py` | ClaudeDebugger class |
| `src/pipeline_monitor/anomaly.py` | 12 anomaly signal detectors |
| `src/pipeline_monitor/change_log.py` | JSONL audit trail writer |
| `src/pipeline_monitor/diagnostic.py` | Diagnostic record builder |
| `src/inference_tap.py` | TapWriter: append-only inference log (prompt/response per LLM call) |
| `src/graph/nodes.py` | REPL tap writers: `_tap_write_repl_exec()`, `_tap_write_repl_result()` |
| `/mnt/raid0/llm/tmp/inference_tap.log` | Inference tap file (created by TUI, deleted on exit) |
| `/mnt/raid0/llm/tmp/repl_tap.log` | REPL tap file (truncated on TUI start, deleted on exit) |
| `logs/debug_changes.jsonl` | Audit trail (fixes + reasoning) |
| `logs/seeding_diagnostics.jsonl` | Per-answer diagnostics |
| `orchestration/prompts/` | Hot-swappable prompt templates |

</details>

---

| [Chapter 25: Cost-Aware Rewards](25-cost-aware-rewards.md) | [Index](INDEX.md) | [Chapter 27: SkillBank Experience Distillation](27-skillbank-experience-distillation.md) |
