#!/usr/bin/env python3
from __future__ import annotations

"""
AMD PACE Benchmark Runner

Tests AMD PACE native PyTorch inference with HuggingFace models.
Compares baseline (no speculation) vs PARD speculative decoding.

Usage:
    # Activate the pace-env first
    conda activate pace-env

    # Run all models
    python bench_amd_pace.py

    # Run specific model
    python bench_amd_pace.py --model qwen7b

    # Run baseline only (no PARD)
    python bench_amd_pace.py --baseline-only

    # Dry run (show what would be tested)
    python bench_amd_pace.py --dry-run

Results are saved to:
    /mnt/raid0/llm/claude/benchmarks/results/runs/{run_id}/amd_pace_{model}_{config}.json
"""

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import torch
from transformers import AutoTokenizer, TextStreamer

from pace.llm import (
    LLMModel,
    SamplingConfig,
    KVCacheType,
    PardSpecDecodeConfig,
)

# Add parent directory for lib imports
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from suites import load_suite, get_all_suite_names

# ============================================================================
# CONFIGURATION
# ============================================================================

# Model configurations
HF_BASE = Path("/mnt/raid0/llm/hf")
RESULTS_DIR = Path("/mnt/raid0/llm/claude/benchmarks/results")

# Set torch threads for AMD EPYC
torch.set_num_threads(96)
torch.set_num_interop_threads(1)

# Default inference settings
TORCH_DTYPE = torch.bfloat16
KV_CACHE = KVCacheType.DYNAMIC
NUM_SPECULATIVE_TOKENS = 10
MAX_NEW_TOKENS = 256

# Model definitions
MODELS = {
    "qwen7b": {
        "target": HF_BASE / "Qwen2.5-7B-Instruct",
        "draft": HF_BASE / "PARD-Qwen2.5-0.5B",
        "family": "qwen",
    },
    "llama8b": {
        "target": HF_BASE / "Llama-3.1-8B-Instruct",
        "draft": HF_BASE / "PARD-Llama-3.2-1B",
        "family": "llama",
    },
    "deepseek32b": {
        "target": HF_BASE / "DeepSeek-R1-Distill-Qwen-32B",
        "draft": HF_BASE / "PARD-DeepSeek-R1-Distill-Qwen-1.5B",
        "family": "qwen",  # Uses Qwen tokenizer
    },
}

# Test prompts for speed comparison
SPEED_TEST_PROMPTS = [
    "Write a Python function that calculates the Fibonacci sequence up to n terms. Include a docstring explaining the function and type hints for all parameters and return value.",
    "Explain the concept of recursion in programming with a simple example.",
    "What are the main differences between Python lists and tuples?",
]


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""
    model_name: str
    config: str  # "baseline" or "pard"
    prompt: str
    response: str
    input_tokens: int
    output_tokens: int
    total_time_sec: float
    tokens_per_second: float
    speculative_stats: Optional[dict] = None
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


@dataclass
class ModelResults:
    """Aggregated results for a model configuration."""
    model_name: str
    config: str
    target_model: str
    draft_model: Optional[str]
    run_id: str
    timestamp: str
    results: list[BenchmarkResult]
    summary: dict[str, Any]


def build_messages(prompt: str, system_prompt: str = "You are a helpful assistant.") -> list[dict]:
    """Build chat messages for the model."""
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]


def build_inputs(tokenizer: AutoTokenizer, prompt: str, system_prompt: str = "You are a helpful assistant."):
    """Build tokenized inputs for the model."""
    messages = build_messages(prompt, system_prompt)

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    encoded = tokenizer.batch_encode_plus(
        [text],
        return_tensors="pt",
        padding="longest",
    )

    return encoded


def load_model(model_path: Path, draft_path: Optional[Path] = None) -> LLMModel:
    """Load AMD PACE model with optional PARD draft."""
    pard_config = None
    if draft_path:
        pard_config = PardSpecDecodeConfig(
            model_name_or_path=str(draft_path),
            num_speculative_tokens=NUM_SPECULATIVE_TOKENS,
        )

    return LLMModel(
        str(model_path),
        dtype=TORCH_DTYPE,
        pard_config=pard_config,
        kv_cache_type=KV_CACHE,
    )


def run_single_benchmark(
    model: LLMModel,
    tokenizer: AutoTokenizer,
    prompt: str,
    model_name: str,
    config: str,
    warmup: bool = False,
) -> Optional[BenchmarkResult]:
    """Run a single benchmark and return results."""
    inputs = build_inputs(tokenizer, prompt)

    gen_kwargs = {
        "max_new_tokens": MAX_NEW_TOKENS,
        "do_sample": False,
        "temperature": 0.0,
        "random_seed": 42,
    }
    sampling_config = SamplingConfig(**gen_kwargs)

    if warmup:
        # Warmup run, not timed
        with torch.inference_mode():
            _ = model.generate(inputs, sampling_config)
        return None

    # Timed run
    start = time.perf_counter()
    with torch.inference_mode():
        outputs = model.generate(inputs, sampling_config)
    elapsed = time.perf_counter() - start

    # Calculate tokens
    output_ids = outputs.output_token_ids[0]
    input_len = inputs["input_ids"].shape[1]
    total_len = output_ids.shape[0]
    new_tokens = total_len - input_len

    tps = new_tokens / elapsed if elapsed > 0 else 0

    # Decode response
    response = tokenizer.decode(output_ids[input_len:], skip_special_tokens=True)

    # Get speculative stats if available
    spec_stats = None
    if hasattr(outputs, 'speculative_stats') and outputs.speculative_stats:
        spec_stats = outputs.speculative_stats

    return BenchmarkResult(
        model_name=model_name,
        config=config,
        prompt=prompt,
        response=response,
        input_tokens=input_len,
        output_tokens=new_tokens,
        total_time_sec=elapsed,
        tokens_per_second=tps,
        speculative_stats=spec_stats,
    )


def run_model_benchmark(
    model_name: str,
    model_config: dict,
    use_pard: bool,
    prompts: list[str],
    run_id: str,
) -> ModelResults:
    """Run full benchmark for a model configuration."""
    target_path = model_config["target"]
    draft_path = model_config["draft"] if use_pard else None
    config_name = "pard" if use_pard else "baseline"

    print(f"\n{'='*60}")
    print(f"Model: {model_name} ({config_name})")
    print(f"Target: {target_path}")
    if draft_path:
        print(f"Draft: {draft_path}")
    print(f"{'='*60}")

    # Check model exists
    if not target_path.exists():
        print(f"ERROR: Target model not found: {target_path}")
        return None
    if draft_path and not draft_path.exists():
        print(f"ERROR: Draft model not found: {draft_path}")
        return None

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(str(target_path))
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    print(f"Loading model (use_pard={use_pard})...")
    model = load_model(target_path, draft_path)

    # Warmup
    print("Warmup run...")
    run_single_benchmark(model, tokenizer, prompts[0], model_name, config_name, warmup=True)

    # Run benchmarks
    results = []
    for i, prompt in enumerate(prompts):
        print(f"\nPrompt {i+1}/{len(prompts)}...")
        result = run_single_benchmark(model, tokenizer, prompt, model_name, config_name)
        if result:
            results.append(result)
            print(f"  Tokens: {result.output_tokens}, Time: {result.total_time_sec:.2f}s, TPS: {result.tokens_per_second:.2f}")
            if result.speculative_stats:
                print(f"  Speculative stats: {result.speculative_stats}")

    # Compute summary
    if results:
        avg_tps = sum(r.tokens_per_second for r in results) / len(results)
        total_tokens = sum(r.output_tokens for r in results)
        total_time = sum(r.total_time_sec for r in results)
    else:
        avg_tps = 0
        total_tokens = 0
        total_time = 0

    summary = {
        "avg_tokens_per_second": avg_tps,
        "total_output_tokens": total_tokens,
        "total_time_sec": total_time,
        "num_prompts": len(prompts),
        "num_results": len(results),
    }

    print(f"\n--- Summary ---")
    print(f"Average TPS: {avg_tps:.2f}")
    print(f"Total tokens: {total_tokens}")
    print(f"Total time: {total_time:.2f}s")

    return ModelResults(
        model_name=model_name,
        config=config_name,
        target_model=str(target_path),
        draft_model=str(draft_path) if draft_path else None,
        run_id=run_id,
        timestamp=datetime.now().isoformat(),
        results=results,
        summary=summary,
    )


def save_results(model_results: ModelResults) -> Path:
    """Save results to JSON file."""
    run_dir = RESULTS_DIR / "runs" / model_results.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    filename = f"amd_pace_{model_results.model_name}_{model_results.config}.json"
    result_path = run_dir / filename

    # Convert to dict
    data = {
        "model_name": model_results.model_name,
        "config": model_results.config,
        "target_model": model_results.target_model,
        "draft_model": model_results.draft_model,
        "run_id": model_results.run_id,
        "timestamp": model_results.timestamp,
        "results": [asdict(r) for r in model_results.results],
        "summary": model_results.summary,
    }

    with open(result_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\nResults saved to: {result_path}")
    return result_path


def get_suite_prompts(suite_names: list[str], max_per_suite: int = 3) -> list[str]:
    """Get prompts from benchmark suites for quality comparison."""
    prompts = []
    for suite_name in suite_names:
        suite = load_suite(suite_name)
        if suite:
            for q in suite.questions[:max_per_suite]:
                prompts.append(q.prompt)
    return prompts


def main():
    parser = argparse.ArgumentParser(description="AMD PACE Benchmark Runner")
    parser.add_argument("--model", choices=list(MODELS.keys()), help="Specific model to test")
    parser.add_argument("--baseline-only", action="store_true", help="Only run baseline (no PARD)")
    parser.add_argument("--pard-only", action="store_true", help="Only run PARD speculation")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be tested")
    parser.add_argument("--quick", action="store_true", help="Quick test with speed prompts only")
    parser.add_argument("--run-id", help="Use specific run ID (default: timestamp)")
    args = parser.parse_args()

    # Determine run ID
    run_id = args.run_id or f"amd_pace_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Determine which models to test
    models_to_test = [args.model] if args.model else list(MODELS.keys())

    # Determine configs (baseline, pard)
    configs = []
    if not args.pard_only:
        configs.append(False)  # baseline
    if not args.baseline_only:
        configs.append(True)   # pard

    # Get prompts
    if args.quick:
        prompts = SPEED_TEST_PROMPTS
    else:
        # Use prompts from thinking and general suites
        prompts = get_suite_prompts(["thinking", "general"], max_per_suite=2)
        if not prompts:
            prompts = SPEED_TEST_PROMPTS

    print(f"AMD PACE Benchmark Runner")
    print(f"Run ID: {run_id}")
    print(f"Models: {models_to_test}")
    print(f"Configs: {['baseline' if not c else 'pard' for c in configs]}")
    print(f"Prompts: {len(prompts)}")

    if args.dry_run:
        print("\n[DRY RUN - No tests will be executed]")
        for model_name in models_to_test:
            config = MODELS[model_name]
            print(f"\n{model_name}:")
            print(f"  Target: {config['target']}")
            print(f"  Draft: {config['draft']}")
            print(f"  Exists: target={config['target'].exists()}, draft={config['draft'].exists()}")
        return

    # Run benchmarks
    all_results = []
    for model_name in models_to_test:
        model_config = MODELS[model_name]
        for use_pard in configs:
            try:
                result = run_model_benchmark(
                    model_name=model_name,
                    model_config=model_config,
                    use_pard=use_pard,
                    prompts=prompts,
                    run_id=run_id,
                )
                if result:
                    save_results(result)
                    all_results.append(result)
            except Exception as e:
                print(f"\nERROR running {model_name} ({'pard' if use_pard else 'baseline'}): {e}")
                import traceback
                traceback.print_exc()

    # Final summary
    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")
    for r in all_results:
        print(f"{r.model_name:15} {r.config:10} {r.summary['avg_tokens_per_second']:>8.2f} t/s")

    print(f"\nResults saved to: {RESULTS_DIR}/runs/{run_id}/")


if __name__ == "__main__":
    main()
