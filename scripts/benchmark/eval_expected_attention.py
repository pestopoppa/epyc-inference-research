#!/usr/bin/env python3
"""Expected Attention KV Cache Compression — RULER NIAH Evaluation.

Evaluates NVIDIA KVPress Expected Attention compression quality on
synthetic Needle-In-A-Haystack retrieval tasks at various retention levels.

Gate criterion (from triattention-kv-selection.md S1):
    >= 90% RULER accuracy at 50% compression

Usage:
    # Dry run (validates pipeline without model):
    python eval_expected_attention.py --dry-run

    # Full run (requires HF model on CPU):
    python eval_expected_attention.py --model Qwen/Qwen2.5-7B-Instruct \\
        --compression-ratios 0.25 0.50 --output-dir results/ea-s1/

Reference: KVPress (NVIDIA), Expected Attention Press
    github.com/NVIDIA/kvpress
"""

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import torch


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def generate_ruler_niah_tasks(
    n_tasks: int = 20,
    context_lengths: list[int] = None,
    seed: int = 42,
) -> list[dict]:
    """Generate synthetic Needle-In-A-Haystack tasks for RULER evaluation.

    Each task embeds a target fact within padding text at a random depth.
    The model must retrieve the fact to answer correctly.
    """
    if context_lengths is None:
        context_lengths = [4096, 8192, 16384]

    import random
    rng = random.Random(seed)

    # Template needles (fact + answer pairs)
    needles = [
        ("The secret code for the vault is 7492.", "7492"),
        ("The capital of Freedonia is Sylvaria.", "Sylvaria"),
        ("Professor Chen's office number is 3817.", "3817"),
        ("The password to the database is quantum-leap-42.", "quantum-leap-42"),
        ("The delivery will arrive at 14:35 on Tuesday.", "14:35"),
        ("Agent Smith's badge number is TK-5519.", "TK-5519"),
        ("The winning lottery numbers are 8, 17, 23, 39, 41.", "8, 17, 23, 39, 41"),
        ("The encryption key is xK9mP2vL7nQ4.", "xK9mP2vL7nQ4"),
        ("The reservation is under the name Fitzgerald.", "Fitzgerald"),
        ("The next meeting is scheduled for Room 204B.", "204B"),
    ]

    # Haystack padding (repeated prose)
    haystack_unit = (
        "The landscape stretched endlessly before them, a vast expanse of rolling hills "
        "covered in golden grass that swayed gently in the warm afternoon breeze. Small "
        "clusters of wildflowers dotted the terrain, adding splashes of purple and white "
        "to the otherwise monochromatic scene. In the distance, a line of mountains rose "
        "against the horizon, their peaks still capped with the last remnants of winter snow. "
    )

    tasks = []
    for i in range(n_tasks):
        ctx_len = rng.choice(context_lengths)
        needle_fact, needle_answer = rng.choice(needles)

        # Build haystack to approximate target token count
        # ~1.3 tokens per word, ~50 words per haystack_unit
        n_units = max(1, ctx_len // 65)
        depth = rng.random()  # 0.0 = start, 1.0 = end
        insert_pos = int(depth * n_units)

        haystack_parts = [haystack_unit] * n_units
        haystack_parts.insert(insert_pos, f"\n{needle_fact}\n")
        context = "".join(haystack_parts)

        question = f"Based on the text above, what is the answer? Retrieve the specific fact."
        tasks.append({
            'id': f'niah_{i:03d}',
            'context': context,
            'question': question,
            'needle': needle_fact,
            'answer': needle_answer,
            'context_length': ctx_len,
            'depth': depth,
        })

    return tasks


# ---------------------------------------------------------------------------
# Expected Attention evaluation
# ---------------------------------------------------------------------------

@dataclass
class EAConfig:
    """Configuration for Expected Attention evaluation."""
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    compression_ratios: list = field(default_factory=lambda: [0.25, 0.50])
    n_future_positions: int = 512
    n_sink: int = 4
    use_covariance: bool = True
    use_vnorm: bool = True
    max_samples: int = 50
    output_dir: str = "benchmarks/results/runs/expected-attention-s1"


@dataclass
class EAResult:
    """Result for a single evaluation sample."""
    sample_id: str
    compression_ratio: float
    # Quality metrics
    correct: Optional[bool] = None        # for scored tasks
    generated_text: str = ""
    reference_answer: str = ""
    # KV cache metrics
    original_kv_entries: int = 0
    compressed_kv_entries: int = 0
    # Timing (prefill + compress + generate are inseparable via KVPress API)
    inference_time_s: float = 0.0


def evaluate_with_expected_attention(
    config: EAConfig,
    dry_run: bool = False,
) -> dict:
    """Run Expected Attention evaluation pipeline.

    Parameters
    ----------
    config : EAConfig
    dry_run : bool
        If True, validates the pipeline without loading a model.

    Returns
    -------
    dict with 'config', 'ruler_results', 'summary'
    """
    print(f"Expected Attention S1 Evaluation")
    print(f"  Model: {config.model_name}")
    print(f"  Compression ratios: {config.compression_ratios}")
    print(f"  Max samples: {config.max_samples}")
    print()

    # Load datasets
    print("Loading datasets...")
    ruler_tasks = generate_ruler_niah_tasks(n_tasks=min(20, config.max_samples))
    print(f"  RULER NIAH tasks: {len(ruler_tasks)}")

    if dry_run:
        print("\n[DRY RUN] Pipeline validated. Skipping model loading.")
        return {
            'config': asdict(config),
            'dry_run': True,
            'ruler_task_count': len(ruler_tasks),
            'status': 'pipeline_validated',
        }

    # --- Load model and tokenizer ---
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading model {config.model_name} ...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        attn_implementation="eager",  # KVPress hooks require eager attention
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"  Model loaded in {time.time() - t0:.1f}s")

    # Set up KVPress Expected Attention
    sys.path.insert(0, "/mnt/raid0/llm/epyc-inference-research/data/external/kvpress")
    from kvpress.presses.expected_attention_press import ExpectedAttentionPress

    def run_with_compression(prompt: str, ratio: float, max_new_tokens: int = 64) -> tuple[str, float]:
        """Run inference with EA compression. Returns (generated_text, inference_time)."""
        press = ExpectedAttentionPress(
            compression_ratio=ratio,
            n_future_positions=config.n_future_positions,
            n_sink=config.n_sink,
            use_covariance=config.use_covariance,
            use_vnorm=config.use_vnorm,
        )
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=16384)
        n_input = inputs["input_ids"].shape[1]

        with torch.no_grad():
            t0 = time.time()
            with press(model):
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                )
            inference_time = time.time() - t0

        generated = tokenizer.decode(outputs[0][n_input:], skip_special_tokens=True)
        return generated, inference_time

    def run_baseline(prompt: str, max_new_tokens: int = 64) -> str:
        """Run inference without compression (baseline)."""
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=16384)
        n_input = inputs["input_ids"].shape[1]
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
            )
        return tokenizer.decode(outputs[0][n_input:], skip_special_tokens=True)

    # --- RULER NIAH evaluation ---
    print("\n=== RULER NIAH Evaluation ===")
    ruler_results = []

    # First run baseline on all RULER tasks
    print("  Running baseline (no compression)...")
    ruler_baseline = {}
    for task in ruler_tasks:
        prompt = f"{task['context']}\n\nQuestion: {task['question']}\nAnswer:"
        baseline_text = run_baseline(prompt)
        correct = task['answer'].lower() in baseline_text.lower()
        ruler_baseline[task['id']] = correct
        print(f"    {task['id']}: baseline={'correct' if correct else 'WRONG'} ({baseline_text[:60]}...)")

    baseline_acc = sum(ruler_baseline.values()) / len(ruler_baseline)
    print(f"  Baseline accuracy: {baseline_acc:.1%} ({sum(ruler_baseline.values())}/{len(ruler_baseline)})")

    # Then run with each compression ratio
    for ratio in config.compression_ratios:
        print(f"\n  Running EA compression_ratio={ratio} ...")
        for task in ruler_tasks:
            prompt = f"{task['context']}\n\nQuestion: {task['question']}\nAnswer:"
            generated, inference_time = run_with_compression(prompt, ratio)
            correct = task['answer'].lower() in generated.lower()

            result = EAResult(
                sample_id=task['id'],
                compression_ratio=ratio,
                correct=correct,
                generated_text=generated[:200],
                reference_answer=task['answer'],
                inference_time_s=inference_time,
            )
            ruler_results.append(result)
            status = "correct" if correct else "WRONG"
            print(f"    {task['id']}: ratio={ratio}, {status} ({generated[:60]}...)")

    # --- Compute summary ---
    print("\n=== Summary ===")
    summary = {'baseline_ruler_acc': baseline_acc}
    gate_passed = False

    for ratio in config.compression_ratios:
        ratio_results = [r for r in ruler_results if r.compression_ratio == ratio]
        acc = sum(1 for r in ratio_results if r.correct) / len(ratio_results) if ratio_results else 0
        avg_time = sum(r.inference_time_s for r in ratio_results) / len(ratio_results) if ratio_results else 0
        summary[f'ruler_acc_{ratio}'] = acc
        summary[f'ruler_avg_time_{ratio}'] = avg_time

        passed = "PASS" if acc >= 0.90 else "FAIL"
        if ratio == 0.50 and acc >= 0.90:
            gate_passed = True
        print(f"  ratio={ratio}: accuracy={acc:.1%} ({sum(1 for r in ratio_results if r.correct)}/{len(ratio_results)}), "
              f"avg_time={avg_time:.2f}s — {passed}")

    summary['gate_passed'] = gate_passed
    gate_str = "GATE PASSED" if gate_passed else "GATE FAILED"
    print(f"\n  S1 Gate (>= 90% RULER at 50% compression): {gate_str}")

    return {
        'config': asdict(config),
        'ruler_results': [asdict(r) for r in ruler_results],
        'summary': summary,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Expected Attention KV Compression Evaluation")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct",
                        help="HuggingFace model name or path")
    parser.add_argument("--compression-ratios", nargs="+", type=float,
                        default=[0.25, 0.50],
                        help="Compression ratios to evaluate (fraction of KV to REMOVE)")
    parser.add_argument("--max-samples", type=int, default=50,
                        help="Max samples per benchmark")
    parser.add_argument("--output-dir", default="benchmarks/results/runs/expected-attention-s1",
                        help="Output directory for results")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate pipeline without loading model")
    args = parser.parse_args()

    config = EAConfig(
        model_name=args.model,
        compression_ratios=args.compression_ratios,
        max_samples=args.max_samples,
        output_dir=args.output_dir,
    )

    results = evaluate_with_expected_attention(config, dry_run=args.dry_run)

    # Save results
    os.makedirs(config.output_dir, exist_ok=True)
    output_file = os.path.join(config.output_dir, "ea_s1_results.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {output_file}")


if __name__ == '__main__':
    main()
