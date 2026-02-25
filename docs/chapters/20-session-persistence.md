# Chapter 20: Session Persistence & Checkpoint/Resume

## Introduction

The session persistence system enables long-running conversations that survive crashes, resume after idle periods, and maintain context across multiple sessions. This is critical for document analysis, iterative benchmarking, and multi-day research tasks.

**Key features:**
- Automatic checkpoints every 5 turns or 30 minutes idle
- Document change detection via SHA-256 hashing
- LLM-extracted findings with confidence scores
- ChromaDB-compatible storage protocol for future semantic search

The system was implemented in 7 phases (2026-01-21 to 2026-01-26) and uses SQLite + numpy for efficient storage.

**Cross-reference: Context compaction** — When session compaction fires (C1 in `src/graph/helpers.py`), full context is externalized to `/mnt/raid0/llm/tmp/session_{id}_ctx_{n}.md` and tracked via `TaskState.context_file_paths`. These files are not managed by the session persistence layer directly but are referenced by the compaction index so the model can `read_file()` them on demand. See Chapter 10 "Session Compaction" for details.

## Architecture Overview

The persistence layer is split into five components, each owning a single responsibility. `SessionPersister` decides *when* to checkpoint, `SQLiteSessionStore` handles *where* the data lands, and `DocumentCache` avoids expensive OCR re-runs. Data classes live in `models.py`, and a CLI wraps everything for interactive use.

<details>
<summary>Component breakdown and storage locations</summary>

| Component | Purpose | Storage |
|-----------|---------|---------|
| `SessionPersister` | Checkpoint triggers & lifecycle | In-memory state |
| `SQLiteSessionStore` | Metadata persistence | `/workspace/orchestration/repl_memory/sessions/sessions.db` |
| `DocumentCache` | OCR result caching | Per-session SQLite: `state/{session_id}/ocr_cache.db` |
| Session models | Data classes | `src/session/models.py` |
| CLI | Session management | `src/cli_sessions.py` |

</details>

<details>
<summary>Storage layout on disk</summary>

```
/workspace/orchestration/repl_memory/sessions/
├── sessions.db                        # Main session metadata (SQLite WAL mode)
├── session_embeddings.npy             # 1024-dim embeddings (TaskEmbedder)
├── state/
│   ├── {session_id}/
│   │   └── ocr_cache.db              # Per-session document cache
│   └── scheduler.json                # Procedure scheduler state
```

</details>

## 7-Phase Implementation

The system was built incrementally over one week. Each phase added a discrete layer, from core data models through to integration tests, so every intermediate state was deployable.

<details>
<summary>Phase-by-phase timeline and files created</summary>

| Phase | Date | Focus | Files Created |
|-------|------|-------|---------------|
| **Phase 1** | 2026-01-21 | Core models & protocol | `models.py`, `protocol.py` |
| **Phase 2** | 2026-01-22 | SQLite store implementation | `sqlite_store.py` |
| **Phase 3** | 2026-01-23 | Document caching layer | `document_cache.py` |
| **Phase 4** | 2026-01-24 | Checkpoint manager | `persister.py` |
| **Phase 5** | 2026-01-25 | CLI interface | `cli_sessions.py` |
| **Phase 6** | 2026-01-25 | API integration | `src/api.py` endpoints |
| **Phase 7** | 2026-01-26 | Testing & validation | `tests/integration/test_sessions.py` |

</details>

## Session Lifecycle

Sessions move through four states based on how long they have been idle. An `ACTIVE` session resumes instantly, while a `STALE` or `ARCHIVED` session triggers document change detection and injects a full context summary so the model can pick up where it left off.

<details>
<summary>Lifecycle states and resume behavior</summary>

| Status | Trigger | Idle Time | Behavior on Resume |
|--------|---------|-----------|-------------------|
| `ACTIVE` | Recent activity | < 1 hour | Direct continuation |
| `IDLE` | No activity | 1 hour - 7 days | Brief context reminder |
| `STALE` | Long idle | 7 - 30 days | Full context injection with document change detection |
| `ARCHIVED` | Manual or 30+ days | > 30 days | "Welcome back" summary, LLM-generated context |

<details>
<summary>Code: Creating a session</summary>

```python
from src.session import SQLiteSessionStore, Session

store = SQLiteSessionStore()

# Create new session
session = Session.create(
    name="Benchmark Analysis",
    project="Model Evaluation",
    working_directory="/mnt/raid0/llm/claude"
)

# Store in database
store.create_session(session)

print(f"Session ID: {session.id}")
print(f"Task ID (for MemRL): {session.task_id}")
```

</details>

<details>
<summary>Code: Tracking session activity</summary>

```python
# Update activity timestamp
session.update_activity()

# Increment message count
session.message_count += 1

# Update topic
session.last_topic = "Analyzing Qwen3-235B benchmark results"

# Save changes
store.update_session(session)
```

</details>

</details>

## Checkpoint System

The `SessionPersister` watches conversation turns and idle time, then writes a checkpoint when thresholds are hit. Each checkpoint captures a SHA-256 hash of the context, serialized artifacts (variables, plots), and the trigger reason. You can also force a checkpoint with the `/save` command.

<details>
<summary>Checkpoint triggers and frequency</summary>

| Trigger | Condition | Frequency |
|---------|-----------|-----------|
| Turn count | Every 5 conversation turns | Common |
| Idle time | 30 minutes without activity | Moderate |
| Explicit save | User `/save` command | Rare |
| Auto-summary | 2 hours idle + no summary exists | Rare |

<details>
<summary>Code: Using the SessionPersister</summary>

```python
from src.session import SessionPersister

persister = SessionPersister(
    session_store=store,
    session_id=session.id,
    llm_summarizer=None,  # Optional LLM function for summaries
    progress_logger=None   # Optional ProgressLogger for MemRL
)

# After each REPL turn
persister.on_turn(repl_env)

# Check if checkpoint needed
if persister.should_checkpoint():
    checkpoint = persister.save_checkpoint(repl_env)
    print(f"Checkpoint saved: {checkpoint.id}")
```

</details>

<details>
<summary>Code: Building a checkpoint manually</summary>

```python
from src.session.models import Checkpoint
import hashlib

# Create checkpoint
checkpoint = Checkpoint(
    id=str(uuid.uuid4()),
    session_id=session.id,
    created_at=datetime.now(timezone.utc),
    context_hash=hashlib.sha256(context_str.encode()).hexdigest(),
    artifacts={
        "variables": {"x": 42, "results": [1, 2, 3]},
        "plots": ["plot_abc123.png"]
    },
    execution_count=45,
    exploration_calls=12,
    message_count=session.message_count,
    trigger="turns"  # or "idle", "explicit", "summary"
)

# Save to store
store.save_checkpoint(checkpoint)
```

</details>

</details>

## Document Tracking & Change Detection

Every document processed during a session gets a SHA-256 fingerprint stored alongside the session record. On resume, the system re-hashes each file and flags anything that has changed or gone missing. OCR results are cached per-session in a dedicated SQLite database so re-processing a 40-page PDF is effectively free.

<details>
<summary>Document registration and OCR caching workflow</summary>

<details>
<summary>Code: Adding a document to a session</summary>

```python
from src.session.models import SessionDocument
from pathlib import Path

# Process a PDF
file_path = Path("/mnt/raid0/llm/docs/whitepaper.pdf")

# Compute file hash
file_hash = SessionDocument.compute_file_hash(file_path)

# Create document record
doc = SessionDocument(
    id=str(uuid.uuid4()),
    session_id=session.id,
    file_path=str(file_path),
    file_hash=file_hash,
    processed_at=datetime.now(timezone.utc),
    total_pages=42,
    cache_path="state/{session_id}/ocr_cache.db"
)

store.add_document(doc)
```

</details>

<details>
<summary>Code: OCR cache lookup and storage</summary>

```python
from src.session.document_cache import DocumentCache

cache = DocumentCache(session_id=session.id, session_store=store)

# Check cache before OCR
cached_result = cache.get_cached("/path/to/document.pdf")

if cached_result:
    print(f"Cache hit! {cached_result.total_pages} pages")
else:
    # Run OCR
    result = run_ocr("/path/to/document.pdf")

    # Cache result
    cache.cache_result("/path/to/document.pdf", result, track_in_session=True)
```

</details>

<details>
<summary>Code: Detecting document changes on resume</summary>

```python
# Build resume context (includes change detection)
resume_ctx = store.build_resume_context(session.id)

# Check for changes
if resume_ctx.document_changes:
    for change in resume_ctx.document_changes:
        if not change.exists:
            print(f"⚠ Document missing: {change.file_path}")
        elif change.new_hash != change.old_hash:
            print(f"⚠ Document changed: {change.file_path}")
```

</details>

</details>

## Findings System

Findings are the key insights extracted during a session. They come from three sources: the user explicitly marking something important, the LLM extracting a claim from conversation context, or a heuristic rule firing. Each finding carries a confidence score and a confirmation flag so you can distinguish verified facts from tentative extractions.

<details>
<summary>Finding sources and confidence levels</summary>

| Source | Confidence | Requires Confirmation |
|--------|------------|-----------------------|
| `USER_MARKED` | 1.0 | No (explicitly marked) |
| `LLM_EXTRACTED` | 0.0-1.0 | Yes (LLM can hallucinate) |
| `HEURISTIC` | 0.7-0.9 | Yes (rule-based) |

<details>
<summary>Code: Creating a user-marked finding</summary>

```python
from src.session.models import Finding, FindingSource

# User explicitly marks a finding
finding = Finding(
    id=str(uuid.uuid4()),
    session_id=session.id,
    content="Qwen3-235B achieves 6.75 t/s with MoE expert reduction to 4",
    source=FindingSource.USER_MARKED,
    created_at=datetime.now(timezone.utc),
    confidence=1.0,
    confirmed=True,
    tags=["benchmark", "moe", "optimization"],
    source_file="/mnt/raid0/llm/claude/benchmarks/results/runs/2026-01-15/..."
)

store.add_finding(finding)
```

</details>

<details>
<summary>Code: LLM-extracted findings with confirmation workflow</summary>

```python
# LLM extracts finding during conversation
llm_finding = Finding(
    id=str(uuid.uuid4()),
    session_id=session.id,
    content="Document suggests using temperature=0.7 for VL models",
    source=FindingSource.LLM_EXTRACTED,
    created_at=datetime.now(timezone.utc),
    confidence=0.75,  # LLM confidence
    confirmed=False,   # Needs user confirmation
    source_page=12
)

store.add_finding(llm_finding)

# User confirms later
llm_finding.confirmed = True
store.update_finding(llm_finding)
```

</details>

</details>

## Resume Context

When a session resumes, the system assembles a structured context block containing the session name, all tracked documents (with change flags), confirmed findings, and the last conversation topic. This block is formatted as markdown and injected as the system message so the model immediately knows what happened before.

<details>
<summary>Resume context assembly and injection</summary>

<details>
<summary>Code: Building and injecting resume context</summary>

```python
# Build context for session resume
context = store.build_resume_context(session.id)

# Context includes:
print(f"Session: {context.session.name}")
print(f"Documents: {len(context.documents)}")
print(f"Findings: {len(context.findings)}")
print(f"Changes detected: {len(context.document_changes)}")

# Format for LLM injection
llm_context = context.format_for_injection()

# Inject at conversation start
messages = [
    {"role": "system", "content": llm_context},
    {"role": "user", "content": "Continue where we left off..."}
]
```

</details>

<details>
<summary>Data: Example injected markdown format</summary>

```markdown
# Session Resumed: Benchmark Analysis
Last active: 2026-01-27 14:30 (47 messages)

## Documents
- /mnt/raid0/llm/docs/whitepaper.pdf (42 pages, processed)
- /mnt/raid0/llm/benchmarks/results/runs/.../results.json (1 pages, CHANGED)

## Key Findings from Previous Session
1. Qwen3-235B achieves 6.75 t/s with MoE expert reduction to 4
2. Prompt lookup provides 12.7x speedup on summarization tasks
3. SSM models (Qwen3-Next) cannot use speculative decoding
... and 7 more findings

## Last Conversation Topic
Analyzing optimal expert count for MoE models

## Warnings
- Source file changed: /mnt/raid0/llm/benchmarks/results/runs/.../results.json
```

</details>

</details>

## CLI Interface

The `cli_sessions.py` module exposes all session operations through the `orch sessions` command group. You can list, search, inspect, resume, archive, and delete sessions from your terminal without touching Python directly.

<details>
<summary>CLI commands and example output</summary>

<details>
<summary>Code: Available CLI commands</summary>

```bash
# List all active sessions
orch sessions list --status active

# Search sessions by topic
orch sessions search "benchmark"

# Show session details
orch sessions show abc123 --findings --checkpoints

# Resume a session (get context injection)
orch sessions resume abc123

# Archive old sessions
orch sessions archive abc123

# Delete a session (requires --force)
orch sessions delete abc123 --force
```

</details>

<details>
<summary>Code: Example list output</summary>

```bash
$ orch sessions list --status active

● abc123  Benchmark Analysis      47 msgs   2h ago   [benchmark, moe]
● def456  Document Formalization  12 msgs   30m ago  [ocr, vision]
○ ghi789  Code Review             8 msgs    1d ago   [code, refactor]

3 sessions found
```

</details>

</details>

## Storage Performance

The storage footprint is intentionally small. A typical session with a handful of documents and checkpoints stays well under 5MB. SQLite WAL mode gives us crash safety and good concurrency without the overhead of a full database server.

<details>
<summary>Per-component storage costs and WAL mode benefits</summary>

| Metric | Value | Notes |
|--------|-------|-------|
| Session metadata | ~2KB/session | SQLite row |
| Checkpoint | ~5-50KB | Depends on artifact count |
| Document record | ~1KB | Reference only, OCR cached separately |
| OCR cache | ~100KB-2MB/doc | Compressed JSON, no images |
| Embeddings | 1024 x 4 bytes = 4.0KB | Per session (TaskEmbedder) |

**WAL mode benefits:**
- Crash-safe (writes to WAL first)
- Better concurrency (readers don't block writers)
- Automatic checkpoint merging on close

</details>

## Cross-Request REPL Globals (Phase 3)

Phase 3 extends checkpoints to persist user-defined REPL globals across separate `/chat` requests when the client opts in with `session_id`.

- `ChatRequest.session_id` enables restore-on-chat-start behavior.
- `REPLEnvironment.checkpoint()` now captures:
  - `user_globals` (JSON-safe user variables only),
  - `variable_lineage` (role/type/save metadata),
  - `skipped_user_globals` (non-serializable values).
- `REPLEnvironment.restore()` rebuilds builtins, then merges checkpoint user globals.
- `SQLiteSessionStore` persists these fields in `checkpoints` with additive migration for existing databases.
- Checkpoint restore now uses an explicit protocol compatibility boundary:
  - `protocol_version` persisted on checkpoint payloads (`v1` current),
  - missing version is treated as legacy `v0` and upgraded,
  - newer versions are downgraded best-effort to REPL restore schema (`version=1`) with unknown fields dropped.
- `/chat` response diagnostics now include `session_persistence.restore_protocol`:
  - `source_version`, `target_version`,
  - `compat_mode` (`exact`, `legacy_upgrade`, `forward_downgrade`),
  - `missing_required_fields`, `dropped_fields`.
- `SessionPersister` applies payload limits:
  - warning at ~50MB,
  - hard cap at ~100MB with oldest-variable eviction.
- Limits are configurable via env:
  - `ORCHESTRATOR_SESSION_PERSISTENCE_CHECKPOINT_GLOBALS_WARN_MB`
  - `ORCHESTRATOR_SESSION_PERSISTENCE_CHECKPOINT_GLOBALS_HARD_MB`
- `ResumeContext.format_for_injection()` now renders a `Variables (from previous request)` section so resumed sessions expose prior derived state without re-derivation.
- HTTP roundtrip validation now includes a passing `/chat` request1→checkpoint-save→request2→restore integration test (`tests/integration/test_chat_pipeline.py::TestChatEndpoint::test_session_restore_roundtrip_repl_globals`).

## References

<details>
<summary>Source files and API endpoints</summary>

- **Session models**: `src/session/models.py`
- **Protocol (abstract interface)**: `src/session/protocol.py`
- **SQLite store**: `src/session/sqlite_store.py`
- **Document cache**: `src/session/document_cache.py`
- **Checkpoint manager**: `src/session/persister.py`
- **CLI**: `src/cli_sessions.py`
- **API integration**: `src/api.py` (POST /sessions, GET /sessions/{id}/resume)

</details>

---

*Previous: [Chapter 19: Procedure Registry](19-procedure-registry.md)* | *Next: [Chapter 21: Benchmarking Framework](21-benchmarking-framework.md)*
