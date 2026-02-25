# TOON Format Evaluation

**Date**: 2026-01-27
**Status**: Phase 3 Complete - Integration Ready for A/B Testing
**Result**: **ADOPT** for tool outputs (40-65% reduction), **REJECT** for grep hits

## Executive Summary

TOON (Token-Oriented Object Notation) was evaluated for orchestrator performance optimization through token reduction. The evaluation found:

| Use Case | Token Reduction | Recommendation |
|----------|----------------|----------------|
| File listings (`_list_dir`) | **64.6%** | ADOPT |
| Tool output JSON | **55-65%** | ADOPT |
| Escalation context | **42.3%** | ADOPT (for >2 attempts) |
| Grep hits | **-18.6%** (worse) | REJECT - Markdown better |

## Phase 1-2 Results: Token Counting

### Character Reduction (Representative Data)

| Test Case | JSON chars | TOON chars | Reduction |
|-----------|------------|------------|-----------|
| File listing (10 files) | 464 | 163 | **64.9%** |
| TaskIR steps (4 steps) | 406 | 148 | **63.5%** |
| Grep results (5 hits) | 521 | 223 | **57.2%** |
| OCR sections (5 sections) | 539 | 241 | **55.3%** |
| Escalation context | 319 | 178 | **44.2%** |
| **TOTAL** | 2249 | 953 | **57.6%** |

### Token Reduction (tiktoken cl100k_base)

| Test Case | JSON tokens | TOON tokens | Reduction |
|-----------|-------------|-------------|-----------|
| File listing (10 files) | 302 | 107 | **64.6%** |
| Grep results (20 hits) | 657 | 295 | **55.1%** |
| Escalation context | 196 | 113 | **42.3%** |
| Large OCR output (12 sections) | 521 | 233 | **55.3%** |
| **TOTAL** | 1676 | 748 | **55.4%** |

### Why TOON Fails for Grep Hits

TOON was initially expected to help with grep hits, but testing showed Markdown is more efficient:

```
# Markdown format (291 chars):
### Search: `abstract`
Line 12: This paper presents a novel approach...
Line 45: The abstract method enables significant speedups...

### Search: `conclusion`
Line 234: In conclusion, we demonstrate 2x throughput gains

# TOON format (345 chars) - WORSE:
excerpts[4]{pattern,source,line,match}:
  abstract,paper.pdf,12,This paper presents...
  abstract,paper.pdf,45,The abstract method...   <- pattern repeated!
  conclusion,paper.pdf,234,In conclusion...
```

**Root cause**: TOON's tabular format repeats the pattern on every row. Markdown's grouped structure (`### Search: pattern`) avoids this repetition.

## Phase 3: Integration

### Files Modified

| File | Change |
|------|--------|
| `src/services/toon_encoder.py` | NEW - TOON encoding utilities (7 functions) |
| `src/repl_environment.py` | Added `use_toon_encoding` config + integrated 3 tools |
| `src/prompt_builders.py` | Added `use_toon_encoding` config flag |
| `pyproject.toml` | Added `toon` optional dependency |
| `tests/unit/test_toon_encoder.py` | NEW - 17 unit tests |

### Integrated REPL Tools

| Tool | Benefit | Status |
|------|---------|--------|
| `_list_dir()` | **64.6%** reduction | ✅ Integrated |
| `_list_procedures()` | ~55% reduction | ✅ Integrated |
| `_recall()` | ~55% reduction | ✅ Integrated |
| `_file_info()` | Minimal (single object) | Not integrated |
| `_grep()` | -18% (worse) | Rejected |

### Usage

```python
# Enable TOON for REPL tool outputs
from src.repl_environment import REPLEnvironment, REPLConfig

config = REPLConfig(use_toon_encoding=True)
repl = REPLEnvironment(context="...", config=config)

# _list_dir now returns TOON for directories with 3+ files:
# path: /mnt/raid0/llm/claude
# files[5]{name,type,size}:
#   main.py,file,1234
#   utils.py,file,567
#   tests,dir,
#   config.yaml,file,89
#   README.md,file,2048
# total: 5
```

### Install TOON Support

```bash
# Optional dependency
uv pip install "hierarchical-orchestrator[toon]"

# Or directly
uv pip install "toon-format @ git+https://github.com/toon-format/toon-python.git"
```

## When to Use TOON

### Good Use Cases (40-65% reduction)

1. **Uniform arrays of objects** - File listings, search results, log entries
2. **Structured tool outputs** - OCR sections, figure metadata, task steps
3. **Escalation context** - Previous attempts, error history

### Poor Use Cases (no benefit or worse)

1. **Grouped data** - Grep hits grouped by pattern (Markdown better)
2. **Non-uniform objects** - Mixed schemas, variable fields
3. **Small arrays** - Less than 3 items (overhead exceeds savings)
4. **Prose content** - No structural savings

## Next Steps

1. **A/B Testing**: Enable `use_toon_encoding=True` on subset of requests
2. **TTFT Measurement**: Measure time-to-first-token impact on 235B model
3. **Monitor Quality**: Track any accuracy regressions in instruction following

## References

- [TOON Spec v3.0](https://github.com/toon-format/spec)
- [Python Implementation](https://github.com/toon-format/toon-python)
- Handoff: `handoffs/active/toon_format_integration.md`
