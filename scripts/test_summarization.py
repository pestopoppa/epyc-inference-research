#!/usr/bin/env python3
from __future__ import annotations

"""Test summarization with different models."""

import json
import time
import httpx
from pathlib import Path

# Load figure descriptions from previous analysis
ANALYSIS_FILE = Path("/mnt/raid0/llm/epyc-inference-research/Twyne_V1_Whitepaper_analysis.json")

def load_context():
    """Load figure descriptions and construct context."""
    with open(ANALYSIS_FILE) as f:
        data = json.load(f)

    # Build context from figures
    context_parts = [
        "# Twyne V1 Whitepaper Analysis",
        "",
        "## Document Overview",
        "This is a technical whitepaper for a DeFi protocol called Twyne.",
        "Total pages: 20",
        "",
        "## Figure Descriptions",
        "",
    ]

    for fig in data.get("figures", []):
        context_parts.append(f"### {fig['id']} (Page {fig['page']})")
        context_parts.append(fig["description"])
        context_parts.append("")

    return "\n".join(context_parts)

SUMMARIZATION_PROMPT = """Based on the document analysis above, write a concise executive summary (300-400 words) covering:

1. Main thesis and purpose of the Twyne protocol
2. Key innovations (credit delegation, dual-LTV framework)
3. How it works (integration with DeFi lending protocols)
4. Benefits for borrowers and lenders
5. Target audience

Write the summary directly, without any thinking or reasoning steps. Be concise and professional."""

def test_summarization(server_url: str, model_name: str):
    """Run summarization test against a server."""
    context = load_context()
    full_prompt = f"{context}\n\n---\n\n{SUMMARIZATION_PROMPT}"

    print(f"Testing {model_name} on {server_url}")
    print(f"Context length: {len(context)} chars")
    print(f"Full prompt length: {len(full_prompt)} chars")
    print()

    start = time.time()

    with httpx.Client(timeout=300) as client:
        response = client.post(
            f"{server_url}/v1/chat/completions",
            json={
                "model": model_name,
                "messages": [{"role": "user", "content": full_prompt}],
                "max_tokens": 800,
                "temperature": 0,
            },
        )

    elapsed = time.time() - start
    result = response.json()

    if "choices" not in result:
        print(f"ERROR: {result}")
        return None

    content = result["choices"][0]["message"]["content"]
    usage = result.get("usage", {})

    # Calculate tokens per second
    completion_tokens = usage.get("completion_tokens", len(content.split()))
    tps = completion_tokens / elapsed if elapsed > 0 else 0

    print(f"Elapsed: {elapsed:.2f}s")
    print(f"Tokens: {completion_tokens}")
    print(f"Speed: {tps:.2f} t/s")
    print()
    print("=" * 60)
    print("SUMMARY OUTPUT:")
    print("=" * 60)
    print(content)
    print("=" * 60)

    return {
        "model": model_name,
        "server": server_url,
        "elapsed_sec": elapsed,
        "tokens": completion_tokens,
        "tokens_per_sec": tps,
        "summary_length": len(content),
        "summary": content,
    }

if __name__ == "__main__":
    import sys

    server_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8091"
    model_name = sys.argv[2] if len(sys.argv) > 2 else "Qwen2.5-Coder-32B"

    result = test_summarization(server_url, model_name)

    if result:
        # Save result
        output_file = Path(f"/mnt/raid0/llm/epyc-inference-research/Twyne_summarization_{model_name.replace('/', '_')}.json")
        with open(output_file, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nResult saved to: {output_file}")
