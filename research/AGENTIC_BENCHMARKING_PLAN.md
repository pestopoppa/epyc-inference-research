# Agentic/Tool-Use Model Benchmarking Plan

**Created:** 2025-12-16
**Status:** Active

---

## Purpose

Benchmark tool-calling and agentic capabilities critical for the orchestration system. All models that participate in orchestration must reliably:
1. Generate valid tool calls with correct schemas
2. Handle tool results appropriately
3. Recover from errors gracefully
4. Maintain state across multi-turn interactions

---

## Critical Importance

**This benchmark is arguably the most important** because orchestration quality depends entirely on reliable tool calling. A model that scores well on coding but generates malformed tool calls is useless as an orchestration participant.

---

## Quality Rubric (Tiered Difficulty)

**Scoring:**
- 1-2: Invalid schema, wrong tool, or broken JSON
- 3: Valid but suboptimal (missing optional fields, verbose)
- 4: Correct with minor style issues
- 5: Correct, minimal, well-structured

---

### TIER 1: Baseline (All orchestration participants must score 4-5)

#### A-T1-Q1: Single Tool Call
```
Prompt: "You have access to a tool:
{
  \"name\": \"get_weather\",
  \"parameters\": {
    \"city\": \"string (required)\",
    \"units\": \"string (optional, 'celsius' or 'fahrenheit', default 'celsius')\"
  }
}

User asks: 'What's the weather in Tokyo?'
Generate the tool call as JSON."

Expected:
{"name": "get_weather", "parameters": {"city": "Tokyo"}}
```
| Criterion | Weight |
|-----------|--------|
| Valid JSON | 30% |
| Correct tool name | 20% |
| Required param present | 30% |
| No hallucinated params | 20% |

#### A-T1-Q2: Tool with Multiple Parameters
```
Prompt: "Tool available:
{
  \"name\": \"search_files\",
  \"parameters\": {
    \"pattern\": \"string (required) - glob pattern\",
    \"directory\": \"string (required) - path to search\",
    \"max_results\": \"integer (optional, default 10)\"
  }
}

User asks: 'Find all Python files in /src with max 5 results'
Generate the tool call."

Expected:
{"name": "search_files", "parameters": {"pattern": "*.py", "directory": "/src", "max_results": 5}}
```
| Criterion | Weight |
|-----------|--------|
| Valid JSON | 25% |
| Correct pattern inference (*.py) | 25% |
| Both required params | 30% |
| Optional param when specified | 20% |

#### A-T1-Q3: Choosing Correct Tool
```
Prompt: "Available tools:
1. read_file: {\"path\": \"string\"} - Read file contents
2. write_file: {\"path\": \"string\", \"content\": \"string\"} - Write to file
3. list_directory: {\"path\": \"string\"} - List directory contents

User asks: 'Show me what's in the config folder'
Which tool and what parameters?"

Expected:
{"name": "list_directory", "parameters": {"path": "config"}}
```
| Criterion | Weight |
|-----------|--------|
| Correct tool selection | 50% |
| Valid parameters | 30% |
| Valid JSON | 20% |

---

### TIER 2: Medium-Hard (worker: 3-4, orchestrator: 4-5)

#### A-T2-Q1: Sequential Tool Calls
```
Prompt: "Tools:
- read_file: {\"path\": \"string\"}
- grep_search: {\"pattern\": \"string\", \"path\": \"string\"}

User asks: 'Find where ERROR is logged in /var/log/app.log, then show me that file'

Generate the tool calls in order as a JSON array."

Expected:
[
  {"name": "grep_search", "parameters": {"pattern": "ERROR", "path": "/var/log/app.log"}},
  {"name": "read_file", "parameters": {"path": "/var/log/app.log"}}
]
```
| Criterion | Weight |
|-----------|--------|
| Both tools included | 30% |
| Correct order | 25% |
| Valid parameters | 25% |
| Valid JSON array | 20% |

#### A-T2-Q2: Handling Tool Results
```
Prompt: "You called: {\"name\": \"get_user\", \"parameters\": {\"id\": 123}}

Tool returned: {\"error\": \"User not found\", \"code\": 404}

Tools available:
- get_user: {\"id\": \"integer\"}
- search_users: {\"query\": \"string\"} - Search by name/email
- create_user: {\"name\": \"string\", \"email\": \"string\"}

What should you do next? Generate your response (tool call or message to user)."

Expected: Either search_users to find correct user, or ask user for more info.
NOT: Retry same call or create_user without asking.
```
| Criterion | Weight |
|-----------|--------|
| Acknowledges error | 25% |
| Reasonable next action | 35% |
| Doesn't repeat failed call | 25% |
| Valid format | 15% |

#### A-T2-Q3: Nested/Complex Parameters
```
Prompt: "Tool:
{
  \"name\": \"create_task\",
  \"parameters\": {
    \"title\": \"string (required)\",
    \"assignees\": \"array of strings (required)\",
    \"metadata\": {
      \"priority\": \"string (low/medium/high)\",
      \"tags\": \"array of strings\",
      \"due_date\": \"string (ISO date)\"
    }
  }
}

User: 'Create a high priority task called \"Fix login bug\" for Alice and Bob, tagged as \"bug\" and \"urgent\", due tomorrow (2024-03-15)'"

Expected:
{
  "name": "create_task",
  "parameters": {
    "title": "Fix login bug",
    "assignees": ["Alice", "Bob"],
    "metadata": {
      "priority": "high",
      "tags": ["bug", "urgent"],
      "due_date": "2024-03-15"
    }
  }
}
```
| Criterion | Weight |
|-----------|--------|
| Valid nested JSON | 25% |
| Arrays correct | 25% |
| All fields populated | 30% |
| Correct data types | 20% |

---

### TIER 3: Hard (worker: 2-3, orchestrator: 3-4, architect: 4-5)

#### A-T3-Q1: Ambiguous Request Resolution
```
Prompt: "Tools:
- send_email: {\"to\": \"string\", \"subject\": \"string\", \"body\": \"string\"}
- send_slack: {\"channel\": \"string\", \"message\": \"string\"}
- create_ticket: {\"title\": \"string\", \"description\": \"string\", \"assignee\": \"string\"}

User: 'Let the team know about the outage'

This is ambiguous. What clarifying question would you ask, OR if you must act,
which tool is most appropriate and why?"

Expected: Either ask for clarification (preferred) OR choose send_slack with reasoning
(team communication, urgent, real-time).
```
| Criterion | Weight |
|-----------|--------|
| Recognizes ambiguity | 30% |
| Asks clarification OR provides reasoning | 35% |
| Reasonable choice if acting | 25% |
| Doesn't blindly pick wrong tool | 10% |

#### A-T3-Q2: Error Recovery Chain
```
Prompt: "Conversation so far:

You: {\"name\": \"deploy\", \"parameters\": {\"env\": \"prod\", \"version\": \"1.2.3\"}}
Tool: {\"error\": \"Deployment blocked: failing health check on staging\"}

You: {\"name\": \"get_health\", \"parameters\": {\"env\": \"staging\"}}
Tool: {\"status\": \"unhealthy\", \"failing_checks\": [\"database_connection\"]}

You: {\"name\": \"check_database\", \"parameters\": {\"env\": \"staging\"}}
Tool: {\"status\": \"connection_refused\", \"host\": \"db-staging.internal\", \"port\": 5432}

What's your next action? Available tools:
- restart_service: {\"service\": \"string\", \"env\": \"string\"}
- get_logs: {\"service\": \"string\", \"env\": \"string\", \"lines\": \"integer\"}
- notify_oncall: {\"message\": \"string\", \"severity\": \"string\"}
- check_dns: {\"hostname\": \"string\"}"

Expected: Logical next step - either check_dns for db-staging.internal,
get_logs for database service, or restart_service for database.
Should show reasoning about the error chain.
```
| Criterion | Weight |
|-----------|--------|
| Understands error chain | 30% |
| Logical next action | 30% |
| Correct parameters | 25% |
| Shows reasoning | 15% |

#### A-T3-Q3: Schema Edge Cases
```
Prompt: "Tool schema:
{
  \"name\": \"query_api\",
  \"parameters\": {
    \"endpoint\": \"string (required) - must start with /\",
    \"method\": \"string (required) - GET/POST/PUT/DELETE\",
    \"body\": \"object (required for POST/PUT, must be null for GET/DELETE)\",
    \"headers\": \"object (optional)\"
  }
}

User: 'GET the users endpoint with an auth header'

Generate the correct tool call."

Expected:
{
  "name": "query_api",
  "parameters": {
    "endpoint": "/users",
    "method": "GET",
    "body": null,
    "headers": {"Authorization": "..."}  // or ask for auth value
  }
}
```
| Criterion | Weight |
|-----------|--------|
| Endpoint starts with / | 20% |
| Method is GET | 20% |
| Body is null (not omitted) | 30% |
| Headers included | 20% |
| Valid JSON | 10% |

#### A-T3-Q4: Multi-Tool Orchestration Plan
```
Prompt: "You need to deploy a hotfix. Tools available:
- git_checkout: {\"branch\": \"string\"}
- run_tests: {\"suite\": \"string\"} - returns pass/fail
- build_image: {\"tag\": \"string\"}
- deploy: {\"env\": \"string\", \"image\": \"string\"}
- notify_slack: {\"channel\": \"string\", \"message\": \"string\"}
- rollback: {\"env\": \"string\", \"to_version\": \"string\"}

Current state: main branch, last deploy was v1.2.2, hotfix is on branch 'hotfix/auth-fix'

Create a deployment plan as ordered tool calls with conditional logic:
- If tests fail, notify and stop
- If deploy fails, rollback and notify
- If success, notify

Output as JSON with structure: {\"steps\": [...], \"on_test_fail\": [...], \"on_deploy_fail\": [...], \"on_success\": [...]}"

Expected: Complete plan with proper sequencing and error handling branches.
```
| Criterion | Weight |
|-----------|--------|
| Correct sequence (checkout→test→build→deploy) | 25% |
| Test failure handling | 20% |
| Deploy failure handling (rollback) | 20% |
| Success notification | 15% |
| Valid JSON structure | 20% |

---

## Rubric Scoring Summary

| Tier | worker Target | orchestrator Target | architect Target |
|------|--------------|---------------------|-----------------|
| T1 (Baseline) | 4-5 | 5 | 5 |
| T2 (Medium) | 3-4 | 4-5 | 5 |
| T3 (Hard) | 2-3 | 3-4 | 4-5 |

**CRITICAL: Minimum Requirements for Orchestration Participation**
- ANY model in orchestration: Must score 4+ on T1
- Orchestrator roles: Must score 4+ on T1-T2
- Architect roles: Must score 4+ on all tiers

---

## Models to Benchmark

All models that participate in orchestration should be tested:
- All worker models
- All escalation models
- All architect models
- All oracle models

### Priority Models
| Model | Role | Notes |
|-------|------|-------|
| Qwen2.5-7B-Instruct | worker_general | Fast, must pass T1 |
| Qwen2.5-Coder-32B | coder_primary | Must handle tool schemas |
| Qwen3-30B-A3B-Instruct | escalation | MoE, orchestrator candidate |
| Qwen3-235B-A22B | architect | Heavy, orchestration lead |

---

## Benchmark Scripts

### Single Model Testing
```bash
# Dense model (baseline only)
./scripts/benchmark/run_agentic_rubric.sh /path/to/model.gguf ModelName dense

# MoE model (baseline + moe4 configurations)
./scripts/benchmark/run_agentic_rubric.sh /path/to/model.gguf ModelName qwen3moe
```

### Full Suite
```bash
# Run all models with appropriate configs
./scripts/benchmark/run_all_agentic_benchmarks.sh
```

### Architecture Types
| Arch | Configurations Tested |
|------|----------------------|
| `dense` | baseline |
| `qwen3moe` | baseline, moe4 |

Results are saved with config prefix: `{model}_{config}_{test}.txt`
Output directory: `/tmp/claude/agentic_rubric_results/`

### Overnight Suite
```bash
# Run all benchmarks (thinking, coder, vl, general, agentic) + optimization tests
./scripts/benchmark/run_overnight_benchmark_suite.sh

# Run only agentic benchmark suite
./scripts/benchmark/run_overnight_benchmark_suite.sh --suite agentic

# Dry run to see what would execute
./scripts/benchmark/run_overnight_benchmark_suite.sh --dry-run
```

---

## Results (fill in after benchmarking)

| Model | Quant | Config | Speed | T1 | T2 | T3 | Orchestration Ready? |
|-------|-------|--------|-------|----|----|-----|---------------------|
| TBD | | | | | | | |

---

## Notes

- JSON validity is critical - even one syntax error = score 1
- Tool name must match exactly (case-sensitive)
- Parameter names must match schema exactly
- Models that struggle with T1 should NOT be used in orchestration
