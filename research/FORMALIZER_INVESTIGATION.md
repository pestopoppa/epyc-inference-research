# Research Handoff: Formalizer Model Investigation

**Created**: 2026-01-06
**Purpose**: Investigate specialized open-source models that can serve as "formalizers" for various task types in the hierarchical orchestration system.

---

## Background

### What is a Formalizer?

A formalizer is a preprocessing model that converts vague, ambiguous task descriptions into well-defined formal specifications. The key insight:

> **Invest tokens upfront to clarify the problem space, saving tokens downstream from failed attempts.**

### Current Implementation

We have implemented a formalizer role using MathSmith-Hard-Problem-Synthesizer-Qwen3-8B for mathematical problem formalization:

- **Model**: `MathSmith-Hard-Problem-Synthesizer-Qwen3-8B.Q8_0` (8.1 GB)
- **Output**: `FormalizationIR` (JSON schema in `orchestration/formalization_ir.schema.json`)
- **Routing**: Triggered when frontdoor detects implicit math structure, optimization keywords, or high ambiguity

### The Gap

MathSmith handles **mathematical** formalization well, but we lack formalizers for:
1. **Code architecture** - System design, component interfaces, data flow
2. **Agentic workflows** - Tool sequences, multi-step planning
3. **Formal verification** - Pre/post conditions, invariants, proofs
4. **Natural language constraints** - Structured requirements extraction

---

## Research Questions

### Primary Questions

1. **Architecture Formalization**: What models can convert vague system requirements into formal architecture specifications (components, interfaces, data flow, constraints)?

2. **Tool/API Formalization**: What models excel at converting natural language into structured tool calls or API sequences?

3. **Verification Formalization**: What models can extract formal verification conditions (pre/post, invariants) from code or requirements?

4. **Multi-Domain Formalizers**: Are there models that can formalize across multiple domains, or do we need specialized formalizers per task type?

### Secondary Questions

5. **Speed vs Quality**: Can smaller (1-3B) formalizers achieve acceptable quality for preprocessing, or do we need 7-8B models?

6. **Quantization Impact**: How does Q4/Q6/Q8 quantization affect formalization quality? (Formalizers output structured JSON, so precision may matter more)

7. **Speculative Acceleration**: Can formalizers themselves be accelerated with draft models, or is their output too structured?

---

## Candidate Models to Investigate

### Code Architecture Formalizers

| Model | Size | Training Focus | Why Investigate |
|-------|------|----------------|-----------------|
| **GraphCodeBERT** | 125M | Code structure graphs | May extract component relationships |
| **UniXcoder** | 125M | Cross-modal code understanding | Handles code↔NL↔AST |
| **CodeT5+** | 220M-16B | Code understanding + generation | Multi-task code comprehension |
| **StarCoder2** | 3B-15B | Code completion | May understand code structure |
| **DeepSeek-Coder-V2** | 16B/236B | Code + instruction following | Strong at structured output |

**Research Task**: Test each model's ability to output structured architecture descriptions given vague requirements.

### Tool/API Formalizers

| Model | Size | Training Focus | Why Investigate |
|-------|------|----------------|-----------------|
| **Gorilla** | 7B | API call generation | Trained specifically on API docs |
| **ToolLLM** | 7B | Tool use reasoning | Multi-tool orchestration |
| **NexusRaven** | 13B | Function calling | Zero-shot function extraction |
| **Functionary** | 7B-70B | Function calling | OpenAI-compatible tool use |
| **xLAM** | 1B-8B | Agentic tool use | Salesforce's agent models |

**Research Task**: Evaluate which model best converts "I want to download a file and process it" → structured tool call sequence.

### Verification Formalizers

| Model | Size | Training Focus | Why Investigate |
|-------|------|----------------|-----------------|
| **DeepSeek-Prover-V2** | 7B/671B | Lean 4 theorem proving | Formal verification native |
| **fm-universe/Dafny** | 7B | Dafny specifications | Pre/post conditions |
| **fm-universe/Coq** | 7B | Coq proofs | Formal verification |
| **fm-universe/TLA+** | 8B | TLA+ specifications | Distributed system verification |
| **fm-universe/ACSL** | 7B | ANSI/ISO C specs | C code verification |
| **Req2LTL** | Custom | NL → Linear Temporal Logic | Requirement formalization |

**Research Task**: Test extraction of formal specifications from code comments and docstrings.

### Quirky/Novel Models

| Model | Size | What Makes It Quirky | Potential Use |
|-------|------|---------------------|---------------|
| **Palmyra-Fin** | 7B | Financial domain | Contract/requirement extraction |
| **CodeLlama-Instruct** | 7B-34B | Code instruction following | May handle architecture prompts |
| **WizardCoder** | 15B | Evol-Instruct on code | Complex code reasoning |
| **Phind-CodeLlama** | 34B | Code search + generation | Architecture search |
| **SQLCoder** | 7B-15B | SQL generation | Schema formalization |
| **Defog-SQLCoder** | 7B | Text-to-SQL | Database schema extraction |
| **LegalBERT** | 110M | Legal text | Contract clause extraction |
| **BioBERT/PubMedBERT** | 110M | Scientific text | Research requirement extraction |

**Research Task**: Identify if domain-specific models can formalize within their domain better than general models.

---

## Evaluation Framework

### Test Cases

For each candidate model, test on these scenarios:

#### Architecture Formalization Test
```
Input: "Build me a chat application that scales"

Expected FormalizationIR:
- Components: [client, load_balancer, api_gateway, chat_service, message_queue, database, cache]
- Interfaces: [REST API, WebSocket, pub/sub]
- Constraints: [horizontal scaling, message ordering, delivery guarantees]
- Data flow: [client→gateway→service→queue→database]
```

#### Tool Formalization Test
```
Input: "Analyze this CSV and make a chart"

Expected FormalizationIR:
- Tools: [read_file, parse_csv, analyze_data, create_chart, save_file]
- Sequence: [read→parse→analyze→chart→save]
- Parameters: [file_path, chart_type, output_format]
- Error handling: [file_not_found, parse_error, invalid_data]
```

#### Verification Formalization Test
```
Input: "This function should never return null for valid input"

Expected FormalizationIR:
- Precondition: input != null && isValid(input)
- Postcondition: result != null
- Invariant: result.type == expected_type
```

### Scoring Criteria

| Criterion | Weight | Description |
|-----------|--------|-------------|
| **Completeness** | 30% | Does output cover all aspects of the problem? |
| **Correctness** | 30% | Are extracted constraints/components accurate? |
| **Parsability** | 20% | Can output be parsed as valid FormalizationIR? |
| **Speed** | 20% | Tokens/second (formalizers should be fast) |

### Minimum Viable Performance

- **Completeness**: >70% of expected fields populated
- **Correctness**: >80% of extracted items are valid
- **Parsability**: 100% valid JSON (hard requirement)
- **Speed**: >10 t/s on EPYC 9655 (formalizers must not bottleneck)

---

## Investigation Priorities

### High Priority (Investigate First)

1. **ToolLLM / Gorilla / NexusRaven** - Tool formalization is immediately useful for agentic workflows
2. **fm-universe/Dafny** - Verification formalization enables formal correctness checking
3. **xLAM-1B** - If a 1B model can formalize tools, it's nearly free preprocessing

### Medium Priority

4. **DeepSeek-Coder-V2-Lite (16B)** - May handle architecture formalization
5. **CodeT5+ (16B)** - Multi-task code understanding
6. **Functionary-Small (7B)** - Alternative tool formalizer

### Low Priority (Exploratory)

7. **Domain-specific models** - Only if we have domain-specific tasks
8. **GraphCodeBERT/UniXcoder** - Small but may lack generation capability

---

## Deliverables

After investigation, produce:

1. **Model Evaluation Matrix** (`research/formalizer_evaluation.md`)
   - Per-model scores on test cases
   - Speed measurements on EPYC 9655
   - Quantization impact analysis

2. **Model Registry Updates** (`orchestration/model_registry.yaml`)
   - Add successful formalizer models with roles
   - Document routing hints for each formalizer type

3. **FormalizationIR Schema Extensions** (if needed)
   - Add fields for architecture formalization
   - Add fields for tool sequence formalization
   - Add fields for verification conditions

4. **Benchmark Suite** (`benchmarks/prompts/v1/formalizer/`)
   - Test cases for each formalization type
   - Ground truth FormalizationIR for comparison

---

## References

### Papers
- [OR-LLM-Agent](https://arxiv.org/html/2503.10009v1) - End-to-end operations research formalization
- [Formal-LLM Framework](https://arxiv.org/html/2402.00798v2) - Integrating formal methods with LLMs
- [MathSmith Paper](https://arxiv.org/html/2508.05592v1) - Problem synthesis for math training
- [From Informal to Formal](https://arxiv.org/abs/2501.16207) - NL to formal specification

### Model Sources
- [fm-universe on HuggingFace](https://huggingface.co/fm-universe) - Formal methods models
- [Gorilla](https://gorilla.cs.berkeley.edu/) - API call generation
- [ToolLLM](https://github.com/OpenBMB/ToolLLM) - Tool learning framework
- [xLAM](https://huggingface.co/Salesforce/xLAM-1b-fc-r) - Salesforce agent models
- [NexusRaven](https://huggingface.co/Nexusflow/NexusRaven-V2-13B) - Function calling model

### Related Files in This Project
- `orchestration/formalization_ir.schema.json` - Current FormalizationIR schema
- `orchestration/task_ir.schema.json` - TaskIR with formalization field
- `orchestration/model_registry.yaml` - Model registry with formalizer role
- `research/ESCALATION_FLOW.md` - Formalization path in escalation flow

---

## Notes for Investigating Agent

1. **Start with tool formalizers** - Most immediately useful for our agentic workflows
2. **Test on real tasks** - Use actual TaskIR examples from `orchestration/examples/`
3. **Measure actual speed** - Use our standard benchmark methodology
4. **Check GGUF availability** - We need GGUF format for llama.cpp
5. **Consider fine-tuning** - If no model fits, we may need to fine-tune one

Good luck with the investigation!
