# Claude Code Opening Prompt for EPYC 9655 Optimization Project

Copy and paste this as your first message when starting Claude Code:

---

## OPENING PROMPT (FULL)

```
# ⛔ CRITICAL: Set storage constraints FIRST
export HF_HOME=/mnt/raid0/llm/cache/huggingface
export TRANSFORMERS_CACHE=/mnt/raid0/llm/cache/huggingface
export PIP_CACHE_DIR=/mnt/raid0/llm/cache/pip
export TMPDIR=/mnt/raid0/llm/tmp
export XDG_CACHE_HOME=/mnt/raid0/llm/claude/cache
export XDG_DATA_HOME=/mnt/raid0/llm/claude/share
export XDG_STATE_HOME=/mnt/raid0/llm/claude/state

# ALL files MUST be on /mnt/raid0/ — NEVER write to /, /home/, /tmp/, /var/

# 1. Load project context
cat /mnt/raid0/llm/claude/CLAUDE.md

# 2. Source logging (MANDATORY for all actions)
source /mnt/raid0/llm/claude/scripts/utils/agent_log.sh
agent_session_start "LLM optimization session"

# 3. Discover all available models
bash /mnt/raid0/llm/claude/scripts/session/session_init.sh

# 4. Check for untested models
cat /mnt/raid0/llm/claude/logs/untested_models.txt

# 5. Load research report summary
head -100 /mnt/raid0/llm/claude/logs/research_report.md 2>/dev/null || echo "No report yet"

Available agents: @sysadmin, @build-engineer, @benchmark-analyst, @safety-reviewer, @model-engineer, @research-engineer

CURRENT RESEARCH PRIORITY (December 2025):
============================================
✅ PRODUCTION: Track 1 (5.9x), Track 2 (21-48%)
🆕 THIS WEEK: Track 8 (Prompt Lookup), Track 6 (SuffixDecoding)
⛔ DEPRECATED: Track 3 (EAGLE) - blocked, pivot to retrieval methods

Full research plan: /mnt/raid0/llm/claude/research/speculative_decoding_research.md

CRITICAL RULES:
1. ⛔ ALL files on /mnt/raid0/ ONLY
2. Log ALL actions via agent_log.sh
3. Test untested models BEFORE other tasks
4. Update research report after EVERY test
5. Max 3 retries on failures, then STOP

Execute session_init.sh and confirm you see the model inventory.
```

---

## ALTERNATIVE: Quick Start

```
# ⛔ CRITICAL: Set storage paths FIRST
export HF_HOME=/mnt/raid0/llm/cache/huggingface
export TMPDIR=/mnt/raid0/llm/tmp
export PIP_CACHE_DIR=/mnt/raid0/llm/cache/pip
export XDG_CACHE_HOME=/mnt/raid0/llm/claude/cache

source /mnt/raid0/llm/claude/scripts/utils/agent_log.sh && agent_session_start "Quick session"
bash /mnt/raid0/llm/claude/scripts/session/session_init.sh
cat /mnt/raid0/llm/claude/CLAUDE.md | head -150

# Current priority: Implement Track 8 (Prompt Lookup) + Track 6 (SuffixDecoding)
# These compound ON TOP of existing 5.9x speedup from Track 1

⛔ NEVER write to /, /home/, /tmp/, /var/ — ALL files on /mnt/raid0/ ONLY
```

---

## SESSION-SPECIFIC PROMPTS

### 🔥 Track 8: Prompt Lookup Implementation (Priority)
```
Today's goal: Implement Prompt Lookup Decoding (Track 8)

This is zero-cost n-gram matching from prompt — compounds with Track 1.
Expected: +50-100% on summarization, document QA, code editing.

Resources:
- Original: https://github.com/apoorvumang/prompt-lookup-decoding
- vLLM: speculative_model="[ngram]"
- HuggingFace: prompt_lookup_num_tokens parameter

Steps:
1. Check if llama.cpp has native support: ./llama-speculative --help | grep -i ngram
2. If not, implement Python wrapper with simple n-gram matching
3. Test on summarization task, compare with Track 1 alone
4. Measure combined effect: Prompt Lookup → Track 1 fallback
```

### 🔥 Track 6: SuffixDecoding Implementation
```
Today's goal: Implement SuffixDecoding (Track 6)

This builds suffix trees from session outputs for agentic workloads.
Expected: 5-10x on SQL generation, multi-agent, repetitive patterns.

Resources:
- Paper: https://suffix-decoding.github.io/ (NeurIPS 2025 Spotlight)
- Reported: 10.4x on AgenticSQL

Steps:
1. Install suffix_trees: pip install suffix_trees --break-system-packages
2. Create SuffixDraftProvider class
3. Build tree from session outputs
4. Integrate as first-priority draft source before Track 1
```

### Track 1+2 Integration Testing
```
Today's goal: Test combined Track 1 + Track 2 on MoE models

Track 1: External draft (5.9x proven)
Track 2: MoE soft mask (21-48% proven)

Combined command:
./llama-speculative -m MOE_MODEL.gguf -md DRAFT.gguf \
  --override-kv ARCH.expert_used_count=int:4 --draft-max 8 -t 96

Test on: Qwen3-VL-30B-A3B with Qwen3_VL_2B draft
Expected: 7x+ combined speedup
```

### Track 7: CAS-Spec (If Needed)
```
Today's goal: Implement CAS-Spec for models without compatible drafts

Use when: DeepSeek-R1 family (no matching draft model)
Expected: 2.3x speedup via layer-skip cascade

Resources:
- Paper: https://arxiv.org/abs/2510.26843 (NeurIPS 2025)
- Related: CLaSp https://arxiv.org/abs/2505.24196

This requires C++ modifications to llama.cpp.
```

---

## KEY PAPERS (Bookmark These)

### Priority Reading (NeurIPS 2025)
| Paper | Link | Use Case |
|-------|------|----------|
| SuffixDecoding | https://suffix-decoding.github.io/ | Agentic 10x |
| CAS-Spec | https://arxiv.org/abs/2510.26843 | Self-draft 2.3x |

### Implementation Resources
| Resource | Link |
|----------|------|
| Prompt Lookup | https://github.com/apoorvumang/prompt-lookup-decoding |
| vLLM Spec Decode | https://blog.vllm.ai/2024/10/17/spec-decode.html |
| All Spec Decode Papers | https://github.com/hemingkx/SpeculativeDecodingPapers |
| CLaSp/SWIFT | https://arxiv.org/abs/2505.24196 |
| LayerSkip | https://arxiv.org/abs/2404.16710 |
| Kangaroo | https://github.com/Equationliu/Kangaroo |
| REST Retrieval | https://arxiv.org/abs/2311.08252 |
| RASD | https://arxiv.org/abs/2503.03434 |
| CS-Drafting | https://arxiv.org/pdf/2312.11462 |

### Deprecated (Reference Only)
| Paper | Link | Reason |
|-------|------|--------|
| EAGLE | https://github.com/SafeAILab/EAGLE | 0% acceptance, blocked |
| Medusa | https://github.com/FasterDecoding/Medusa | Training required |

---

## COMBINED OPTIMIZATION STACK

```
┌─────────────────────────────────────────────────────────────────┐
│                    DRAFT TOKEN SOURCE                            │
├─────────────────────────────────────────────────────────────────┤
│  1. Prompt Lookup (Track 8) ← TRY FIRST (zero cost)             │
│     └── If n-gram match in prompt → use as draft                │
│                                                                  │
│  2. SuffixDecoding (Track 6) ← TRY SECOND (agentic)             │
│     └── If pattern match in session history → use as draft      │
│                                                                  │
│  3. External Draft Model (Track 1) ← FALLBACK (5.9x)            │
│     └── Qwen2.5-0.5B generates drafts                           │
└─────────────────────────────────────────────────────────────────┘
                              +
┌─────────────────────────────────────────────────────────────────┐
│              TARGET MODEL OPTIMIZATION (Orthogonal)              │
├─────────────────────────────────────────────────────────────────┤
│  Track 2: MoE Soft Mask (+21-48% on MoE models)                 │
│  └── --override-kv ARCH.expert_used_count=int:4                 │
└─────────────────────────────────────────────────────────────────┘

Expected Combined Performance:
- Summarization: 8-12x (Prompt Lookup dominates)
- Agentic/SQL: 10-15x (SuffixDecoding dominates)
- Code generation: 6-7x (Track 1 + Prompt Lookup)
- MoE models: 7x+ (Track 1 + Track 2)
```

---

## SCRIPTS DIRECTORY STRUCTURE

```
/mnt/raid0/llm/claude/scripts/
├── benchmark/
│   ├── bench_zen5.sh
│   ├── run_inference.sh
│   └── record_test.sh
├── session/
│   ├── session_init.sh
│   ├── claude_safe_start.sh
│   ├── health_check.sh
│   ├── monitor_storage.sh
│   └── emergency_cleanup.sh
├── system/
│   └── system_audit.sh
└── utils/
    ├── agent_log.sh
    └── agent_log_analyze.sh
```

---

## CONTEXT WINDOW COMPACTION PROTOCOL

**When the agent needs to compact context, it MUST first:**

```bash
# 1. Log session state
agent_task_start "Pre-compaction save" "Preserving state"
agent_observe "completed_tasks" "List what was done"
agent_observe "current_task" "What's in progress"
agent_observe "pending_tasks" "What remains"

# 2. Update research report with any new findings

# 3. Create summary for retention
echo "=== COMPACTION SUMMARY ===" 
echo "Session: $AGENT_SESSION_ID"
echo "Progress: [describe]"
echo "Next steps: [describe]"
echo "Active models: [list paths]"

agent_task_end "Pre-compaction save" "ready"
```

---

## VERIFYING AGENT BEHAVIOR

```bash
# Summary of activity
/mnt/raid0/llm/claude/scripts/utils/agent_log_analyze.sh --summary

# Detect loops
/mnt/raid0/llm/claude/scripts/utils/agent_log_analyze.sh --loops

# Show errors
/mnt/raid0/llm/claude/scripts/utils/agent_log_analyze.sh --errors

# Get rollback commands
/mnt/raid0/llm/claude/scripts/utils/agent_log_analyze.sh --rollbacks
```
