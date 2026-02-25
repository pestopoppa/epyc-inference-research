# Benchmarking Guide

How to benchmark models on this system.

## Before You Start

1. Ensure model is downloaded to `/mnt/raid0/llm/models/`
2. Test basic launch first (see below)
3. Check [QUIRKS.md](../reference/models/QUIRKS.md) for known issues

## Step 1: Test Basic Launch

Before benchmarking, verify the model runs:

```bash
/mnt/raid0/llm/llama.cpp/build/bin/llama-cli \
  -m /mnt/raid0/llm/models/YOUR_MODEL.gguf \
  -p "Hello, world" -n 10
```

Look for:
- Does it hang? (interactive mode issue)
- Garbage output? (quantization issue)
- Error messages? (compatibility issue)

## Step 2: Run Benchmark Suite

```bash
# All suites
./scripts/benchmark/run_overnight_benchmark_suite.sh --suite all

# Specific suite
./scripts/benchmark/run_overnight_benchmark_suite.sh --suite thinking
./scripts/benchmark/run_overnight_benchmark_suite.sh --suite coder
```

## Step 3: Score with Claude-as-Judge

Results are in `benchmarks/results/runs/`. Score using the rubric:

| Score | Meaning |
|-------|---------|
| 3 | Correct with good reasoning |
| 2 | Partially correct or truncated |
| 1 | Wrong but reasonable attempt |
| 0 | Wrong, empty, or garbage |

## Step 4: Document Results

1. Add to `docs/reference/benchmarks/RESULTS.md`
2. Update `orchestration/model_registry.yaml` with:
   - `baseline_tps`
   - `optimized_tps`
   - Any quirks discovered

## Common Issues

### Model Hangs

Use these flags to prevent interactive mode:
```bash
llama-cli -m MODEL.gguf -f prompt.txt -n 128 \
    --no-display-prompt \
    --simple-io \
    --no-warmup \
    --temp 0
```

### Low Acceptance Rate (<50%)

For speculative decoding:
1. Check draft model compatibility
2. Try temperature 0.3-0.7
3. Reduce K value

### Garbage Output with MoE

Too few experts. Try:
- 4 experts (usually safe)
- 3 experts (test quality)
- Never 2 experts

## The 8 Benchmark Suites

| Suite | Tests | Key For |
|-------|-------|---------|
| Thinking | Reasoning, logic | Oracle roles |
| Coder | Code generation | Coder roles |
| Math | Mathematical proofs | Math workers |
| General | Instruction following | General workers |
| Agentic | Tool calling | Orchestrator |
| VL | Vision-language | Vision workers |
| Long Context | 4K-50K retrieval | Ingestion |
| Instruction Precision | Format compliance | **Critical for orchestration** |

## Results Location

```
benchmarks/
├── prompts/v1/          # Test cases
└── results/
    ├── runs/            # Raw outputs
    ├── reviews/         # Claude-as-Judge scores
    │   ├── {model}_baseline.csv
    │   └── summary.csv
    └── index.jsonl      # Comparison index
```

---

*See [Chapter 21](../chapters/21-benchmarking-framework.md) for methodology details.*
