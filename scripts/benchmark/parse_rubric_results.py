#!/usr/bin/env python3
from __future__ import annotations

"""Parse rubric results from tmp and extract key metrics for scoring."""
import os
import re
import sys
from pathlib import Path

def parse_result_file(filepath):
    """Parse a rubric result file and extract key info."""
    with open(filepath, 'r') as f:
        content = f.read()

    # Extract question (between "user" and "assistant")
    user_match = re.search(r'^user\n(.*?)^assistant', content, re.MULTILINE | re.DOTALL)
    question = user_match.group(1).strip() if user_match else ""

    # Extract response (after "assistant" until "> EOF by user")
    assistant_match = re.search(r'^assistant\n(.*?)> EOF by user', content, re.MULTILINE | re.DOTALL)
    response = assistant_match.group(1).strip() if assistant_match else ""

    # Extract tokens per second
    tps_match = re.search(r'eval time\s*=.*?\(\s*([\d.]+)\s*tokens per second\)', content)
    tps = float(tps_match.group(1)) if tps_match else 0.0

    # Extract prompt tokens
    prompt_match = re.search(r'prompt eval time\s*=.*?/\s*(\d+)\s*tokens', content)
    prompt_tokens = int(prompt_match.group(1)) if prompt_match else 0

    # Extract generated tokens (runs)
    eval_match = re.search(r'eval time\s*=.*?/\s*(\d+)\s*runs', content)
    gen_tokens = int(eval_match.group(1)) if eval_match else 0

    return {
        'question': question,
        'response': response,
        'tps': tps,
        'prompt_tokens': prompt_tokens,
        'gen_tokens': gen_tokens
    }

def list_configs(results_dir):
    """List all unique configs in the results directory."""
    configs = set()
    for f in Path(results_dir).glob('*.txt'):
        # Extract config name (everything before _t[123]_q)
        match = re.match(r'(.+)_t[123]_q', f.name)
        if match:
            configs.add(match.group(1))
    return sorted(configs)

def list_questions_for_config(results_dir, config):
    """List all questions for a given config."""
    questions = []
    for f in Path(results_dir).glob(f'{config}_t*.txt'):
        # Extract question ID
        match = re.search(r'_t(\d+)_q(\d+)_(.+)\.txt$', f.name)
        if match:
            questions.append({
                'file': str(f),
                'tier': int(match.group(1)),
                'qnum': int(match.group(2)),
                'topic': match.group(3),
                'id': f't{match.group(1)}_q{match.group(2)}_{match.group(3)}'
            })
    return sorted(questions, key=lambda x: (x['tier'], x['qnum']))

if __name__ == '__main__':
    results_dir = '/mnt/raid0/llm/tmp/thinking_rubric_results'

    if len(sys.argv) > 1 and sys.argv[1] == 'list':
        print("=== Available configs ===")
        for config in list_configs(results_dir):
            qs = list_questions_for_config(results_dir, config)
            print(f"{config}: {len(qs)} questions")
    elif len(sys.argv) > 2 and sys.argv[1] == 'show':
        config = sys.argv[2]
        for q in list_questions_for_config(results_dir, config):
            result = parse_result_file(q['file'])
            print(f"\n{'='*60}")
            print(f"Question: {q['id']}")
            print(f"TPS: {result['tps']:.1f}")
            print(f"Q: {result['question'][:200]}...")
            print(f"A: {result['response'][:500]}...")
    else:
        print("Usage:")
        print("  python3 parse_rubric_results.py list")
        print("  python3 parse_rubric_results.py show <config_name>")
