# Chapter 19: Procedure Registry & Self-Management

## Introduction

The Procedure Registry provides deterministic self-management for common orchestration tasks. Instead of using 3000-5000 tokens for an LLM to figure out how to benchmark a model, the system executes a pre-defined YAML procedure that costs ~350 tokens.

**Key benefit:** Reduce token usage by 10x for routine tasks while maintaining correctness through schema validation and rollback support.

The registry consists of:
- **ProcedureRegistry** - Loads and validates YAML procedures
- **ProcedureScheduler** - Handles execution with dependency tracking
- **11 YAML procedures** - Covering model lifecycle, benchmarking, and quality gates
- **JSON Schema validation** - Ensures procedure correctness

## Architecture

The registry is built around five components: a `ProcedureRegistry` that loads and validates YAML procedures, a `ProcedureScheduler` that resolves dependencies and runs jobs, the procedure YAML files themselves, a JSON Schema for validation, and a state directory for checkpointing. Each procedure follows a declarative format with metadata, permissions, typed inputs, steps with failure handling, and optional rollback.

<details>
<summary>Core components and procedure format</summary>

### Core Components

| Component | Purpose | Location |
|-----------|---------|----------|
| `ProcedureRegistry` | Load/validate/execute procedures | `orchestration/procedure_registry.py` |
| `ProcedureScheduler` | Dependency-aware execution | `orchestration/procedure_scheduler.py` |
| Procedure YAMLs | Declarative task definitions | `orchestration/procedures/*.yaml` |
| Schema | JSON Schema for validation | `orchestration/procedure.schema.json` |
| State directory | Execution state/checkpoints | `orchestration/procedures/state/` |

### Procedure Definition Format

Every procedure is a YAML file with the following structure:

<details>
<summary>Code: procedure YAML schema</summary>

```yaml
id: benchmark_new_model
name: Benchmark New Model
version: "1.0.0"
description: Run comprehensive benchmark suite
category: benchmark
estimated_tokens: 400

permissions:
  roles: ["frontdoor", "coder_escalation", "admin"]
  requires_approval: false
  destructive: false

inputs:
  - name: model_path
    type: path
    required: true
    validation:
      path_must_exist: true
      path_prefix: "/mnt/raid0/llm/"

outputs:
  - name: results_path
    type: path
  - name: score_pct
    type: float

steps:
  - id: S1
    name: Health Check
    action:
      type: shell
      command: "llama-cli -m ${input.model_path} -p Hello -n 10"
      timeout_seconds: 60
    on_failure: abort

verification:
  gates: ["file_exists", "output_valid"]

rollback:
  enabled: false
```

</details>
</details>

## Available Procedures

The system ships with 11 pre-defined procedures across three categories. Six handle the model lifecycle -- from benchmarking a new model through deprecation. Two cover quality gates and handoff creation. Three more are planned for fine-tuning workflows including dataset prep, training, and evaluation.

<details>
<summary>Procedure catalog by category</summary>

### Model Lifecycle

| Procedure | Purpose | Inputs | Estimated Tokens |
|-----------|---------|--------|------------------|
| `benchmark_new_model` | Run full benchmark suite | model_path, suite | 400 |
| `check_draft_compatibility` | Validate draft-target pairing | draft_path, target_path | 200 |
| `add_model_to_registry` | Add model to registry | model_name, path, role | 250 |
| `update_registry_performance` | Update benchmark results | model_name, results_path | 200 |
| `add_model_quirks` | Document model quirks | model_name, quirk_description | 150 |
| `deprecate_model` | Mark model as deprecated | model_name, reason | 100 |

### Quality & Operations

| Procedure | Purpose | Inputs | Estimated Tokens |
|-----------|---------|--------|------------------|
| `run_quality_gates` | Execute gate sequence | artifact_path, gate_list | 300 |
| `create_handoff` | Create handoff document | topic, blocked_by | 250 |

### Fine-tuning (Planned)

| Procedure | Purpose | Inputs | Estimated Tokens |
|-----------|---------|--------|------------------|
| `prepare_finetuning_dataset` | Generate training data | source_dir, task_type | 500 |
| `run_finetuning` | Execute training run | dataset_path, base_model | 400 |
| `evaluate_finetuned_model` | Benchmark finetuned model | model_path | 400 |

</details>

## ProcedureRegistry Usage

You interact with the registry through a straightforward Python API: initialize it (which auto-loads all YAML files from `orchestration/procedures/`), list available procedures, and call `execute()` with the procedure ID and its inputs. Steps run sequentially with configurable retry and failure handling, and inputs are validated against their schema constraints before any step executes.

<details>
<summary>Registry API and execution details</summary>

### Loading and Executing

<details>
<summary>Code: registry initialization and execution</summary>

```python
from orchestration.procedure_registry import ProcedureRegistry

# Initialize registry (loads all procedures from orchestration/procedures/)
registry = ProcedureRegistry()

# List available procedures
procedures = registry.list()
for proc in procedures:
    print(f"{proc.id}: {proc.description} (~{proc.estimated_tokens} tokens)")

# Execute a procedure
result = registry.execute(
    "benchmark_new_model",
    model_path="/mnt/raid0/llm/models/Qwen2.5-7B-Q4_K_M.gguf",
    model_name="Qwen2.5-7B-Instruct",
    suite="thinking"
)

# Check result
if result.success:
    print(f"Benchmark complete: {result.outputs['score_pct']:.1f}%")
    print(f"Results saved to: {result.outputs['results_path']}")
else:
    print(f"Benchmark failed: {result.error}")
```

</details>

### Step Execution & Retries

Each step has configurable retry and failure handling:

<details>
<summary>Code: step configuration with retries</summary>

```yaml
steps:
  - id: S3
    name: Run Benchmark Suite
    action:
      type: shell
      command: "./scripts/benchmark/run_overnight_benchmark_suite.sh ..."
      timeout_seconds: 3600
    on_failure: retry  # Options: abort, retry, continue, rollback
    max_retries: 2
    depends_on: ["S1", "S2"]  # Execute after S1 and S2 complete
```

</details>

**Failure modes:**
- `abort` - Stop execution immediately
- `retry` - Retry step up to max_retries
- `continue` - Log error but continue to next step
- `rollback` - Execute rollback steps and abort

### Input Validation

Inputs are validated against schema before execution:

<details>
<summary>Code: input validation schema</summary>

```yaml
inputs:
  - name: model_path
    type: path
    required: true
    validation:
      path_must_exist: true
      path_prefix: "/mnt/raid0/llm/"
      pattern: ".*\\.gguf$"

  - name: suite
    type: string
    required: false
    default: "all"
    validation:
      enum: ["all", "thinking", "coder", "general"]

  - name: threads
    type: integer
    required: false
    default: 96
    validation:
      min: 1
      max: 192
```

</details>

**Validation errors prevent execution:**

<details>
<summary>Code: validation error example</summary>

```python
result = registry.execute(
    "benchmark_new_model",
    model_path="/root/model.gguf"  # Not on /mnt/raid0/
)
# Raises ProcedureValidationError: Input 'model_path' must start with '/mnt/raid0/llm/'
```

</details>
</details>

## ProcedureScheduler

The scheduler sits on top of the registry and adds dependency resolution, priority-based ordering, and persistent state. You schedule multiple procedures as jobs, wire them together with `depends_on`, and call `run_all()` to execute the entire DAG. The scheduler saves its state to disk so it can resume after a restart.

<details>
<summary>Scheduler API and state management</summary>

### Scheduling with Dependencies

<details>
<summary>Code: scheduling a procedure chain</summary>

```python
from orchestration.procedure_scheduler import ProcedureScheduler

scheduler = ProcedureScheduler(registry)

# Schedule a chain of procedures
job1 = scheduler.schedule(
    "benchmark_new_model",
    model_path="/path/to/model.gguf",
    priority=10
)

job2 = scheduler.schedule(
    "update_registry_performance",
    depends_on=[job1],  # Wait for benchmark to complete
    model_name="NewModel",
    priority=5
)

job3 = scheduler.schedule(
    "add_model_quirks",
    depends_on=[job2],  # Wait for registry update
    model_name="NewModel",
    quirk_description="Requires temp=0.7 for spec decode"
)

# Execute all scheduled procedures
results = scheduler.run_all()

for job_id, result in results.items():
    print(f"{job_id}: {'✓' if result.success else '✗'}")
```

</details>

### Job Status & Monitoring

<details>
<summary>Code: job status queries</summary>

```python
# Check job status
status = scheduler.get_job_status(job1)
print(f"Status: {status.status}")  # PENDING, RUNNING, COMPLETED, FAILED, BLOCKED

# Get execution progress
progress = scheduler.get_progress()
print(f"Completed: {progress['completed']}/{progress['total']}")
print(f"Running: {progress['running']}")
print(f"Blocked: {progress['blocked']}")
```

</details>

### Persistent State

The scheduler persists state to survive restarts:

<details>
<summary>Code: state persistence</summary>

```python
# State is automatically saved to:
# /mnt/raid0/llm/claude/orchestration/procedures/state/scheduler.json

# After restart, reload state
scheduler = ProcedureScheduler(registry, persist_state=True)
# Automatically resumes pending jobs
```

</details>
</details>

## Approval Workflows

Destructive or sensitive procedures can require explicit approval before they execute. Mark a procedure with `requires_approval: true` and `destructive: true` in its permissions block, and the scheduler will pause execution and return an `ApprovalRequired` checkpoint. You then call `registry.approve()` with the checkpoint ID to resume.

<details>
<summary>Approval flow and permissions</summary>

<details>
<summary>Config: approval permissions block</summary>

```yaml
permissions:
  roles: ["admin"]
  requires_approval: true
  destructive: true  # Deletes data
```

</details>

<details>
<summary>Code: approval execution flow</summary>

```python
result = registry.execute("deprecate_model", model_name="OldModel")

# Scheduler will pause and return:
# ApprovalRequired: Procedure 'deprecate_model' requires approval.
#   Reason: Destructive operation (deletes model files)
#   To approve: registry.approve(job_id)

# Approve the operation
registry.approve(result.checkpoint_id)

# Execution resumes automatically
```

</details>
</details>

## Rollback Support

When a step fails with `on_failure: rollback`, the registry executes the procedure's rollback steps in order -- typically restoring backups and cleaning up partial outputs. You can also trigger rollback manually via `scheduler.rollback(job_id)`. Unhandled exceptions during execution also trigger the rollback path automatically.

<details>
<summary>Rollback configuration and triggers</summary>

<details>
<summary>Config: rollback steps definition</summary>

```yaml
rollback:
  enabled: true
  steps:
    - id: R1
      name: Restore Registry Backup
      action:
        type: shell
        command: "cp model_registry.yaml.bak model_registry.yaml"
    - id: R2
      name: Delete Partial Results
      action:
        type: shell
        command: "rm -rf benchmarks/results/runs/${timestamp}/"
```

</details>

**Rollback is triggered on:**
- Step with `on_failure: rollback`
- Unhandled exception during execution
- Manual rollback via `scheduler.rollback(job_id)`

</details>

## Token Savings

Procedures cut token usage by roughly 88% compared to having the LLM reason through each task from scratch. For a workload of 100 models per year, that translates to saving 835,000 tokens annually -- dropping from 950,000 to 115,000.

<details>
<summary>Token savings breakdown</summary>

**Example comparison:**

| Task | Manual (LLM) | Procedure | Savings |
|------|--------------|-----------|---------|
| Benchmark new model | 3500 tokens | 400 tokens | 88% |
| Add to registry | 2000 tokens | 250 tokens | 87% |
| Check draft compatibility | 1500 tokens | 200 tokens | 86% |
| Run quality gates | 2500 tokens | 300 tokens | 88% |

**Annual savings (100 models/year):**
- Manual: 100 x 9500 tokens = 950,000 tokens
- Procedures: 100 x 1150 tokens = 115,000 tokens
- **Savings: 835,000 tokens (88%)**

</details>

## References

<details>
<summary>File and resource references</summary>

- **ProcedureRegistry**: `orchestration/procedure_registry.py`
- **ProcedureScheduler**: `orchestration/procedure_scheduler.py`
- **Procedure YAMLs**: `orchestration/procedures/*.yaml` (11 files)
- **Schema**: `orchestration/procedure.schema.json`
- **State directory**: `orchestration/procedures/state/`

</details>

---

*Previous: [Chapter 18: Escalation & Routing](18-escalation-and-routing.md)* | *Next: [Chapter 20: Session Persistence](20-session-persistence.md)*
