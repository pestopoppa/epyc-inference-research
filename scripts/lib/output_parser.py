#!/usr/bin/env python3
from __future__ import annotations

"""
Output Parser for LLM Inference Results

Parses llama.cpp output to extract:
- Model response text
- Timing information (tokens/second)
- Acceptance rate (for speculative decoding)
- Token counts

This module is shared with the orchestrator project.
"""

import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class ParsedOutput:
    """Parsed inference output."""

    response: str
    tokens_per_second: Optional[float] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_time_ms: Optional[float] = None
    acceptance_rate: Optional[float] = None
    prompt_eval_time_ms: Optional[float] = None


def parse_response(raw_output: str) -> str:
    """Extract the model's response text from raw output.

    Args:
        raw_output: Raw output from llama.cpp.

    Returns:
        Cleaned response text.
    """
    lines = raw_output.split("\n")
    response_lines = []
    in_response = False
    skip_patterns = [
        "llama_",
        "main:",
        "load_",
        "print_info:",
        "common_",
        "system_info:",
        "sampler",
        "generate:",
        "model:",
        "build:",
        "======",
        "GGUF",
        "llm_load",
        "sampling",
        "[2025",  # timestamp logs
        "timings:",
        "eval time",
        "load time",
        "KV self",
        "Log start",
        "Log end",
        "key-value",
        "tensors",
        "ggml_",
        "Using",
        "Running",
        "encoder_",
        "decoder_",
        "n_ctx",
        "n_batch",
        "n_seq",
        "flash_attn",
        "causal_attn",
        "graph nodes",
        "graph splits",
        "CPU",
        "warming up",
        "total time",
        "unaccounted",
        "graphs reused",
        "memory breakdown",
    ]

    for line in lines:
        # Skip empty lines at start
        if not in_response and not line.strip():
            continue

        # Skip llama.cpp log lines
        skip = False
        for pattern in skip_patterns:
            if pattern in line:
                skip = True
                break

        if skip:
            continue

        # Check for common perf print which signals end of response
        if "common_perf_print" in line:
            break

        # Found response content
        in_response = True
        response_lines.append(line)

    # Join and clean up
    response = "\n".join(response_lines).strip()

    # Remove any remaining prefixes (like "> EOF" markers)
    response = re.sub(r"^> EOF.*$", "", response, flags=re.MULTILINE).strip()

    return response


def parse_timing(raw_output: str) -> dict[str, Optional[float]]:
    """Extract timing information from raw output.

    Returns dict with:
        - tokens_per_second: float or None
        - total_time_ms: float or None
        - prompt_eval_time_ms: float or None
    """
    result = {
        "tokens_per_second": None,
        "total_time_ms": None,
        "prompt_eval_time_ms": None,
    }

    # Check for llama-speculative output format first
    # This has "n_predict = X" and separate "draft:" / "target:" sections
    # The correct speed is: n_predict / (target total time)
    n_predict_match = re.search(r"n_predict\s*=\s*(\d+)", raw_output)
    if n_predict_match:
        n_predict = int(n_predict_match.group(1))

        # Find the target section's total time
        # Look for "target:" marker, then find "total time = X ms" after it
        target_section = re.search(r"target:\s*(.*)", raw_output, re.DOTALL)
        if target_section:
            target_text = target_section.group(1)
            # Pattern: "total time = X ms / Y tokens"
            total_time_match = re.search(
                r"total time\s*=\s*([\d.]+)\s*ms", target_text
            )
            if total_time_match:
                total_time_ms = float(total_time_match.group(1))
                result["total_time_ms"] = total_time_ms
                # Calculate effective speed: tokens / seconds
                if total_time_ms > 0:
                    result["tokens_per_second"] = n_predict / (total_time_ms / 1000)

        # If we found speculative output, also get prompt eval from target section
        if target_section:
            prompt_match = re.search(
                r"prompt eval time\s*=\s*([\d.]+)\s*ms", target_section.group(1)
            )
            if prompt_match:
                result["prompt_eval_time_ms"] = float(prompt_match.group(1))

        # Return early if we successfully parsed speculative output
        if result["tokens_per_second"] is not None:
            return result

    # Standard llama-cli output format
    # Pattern for eval time (NOT prompt eval): "eval time = X ms / Y runs (Z ms per token, W tokens per second)"
    # Must NOT be preceded by "prompt"
    eval_pattern = r"(?<!prompt\s)eval time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*(?:runs|tokens).*?([\d.]+)\s*tokens per second"
    match = re.search(eval_pattern, raw_output)
    if match:
        result["tokens_per_second"] = float(match.group(3))
        result["total_time_ms"] = float(match.group(1))

    # Pattern for prompt eval time (specifically with "prompt" prefix)
    prompt_pattern = r"prompt eval time\s*=\s*([\d.]+)\s*ms"
    match = re.search(prompt_pattern, raw_output)
    if match:
        result["prompt_eval_time_ms"] = float(match.group(1))

    # Alternative pattern: "decoded X tokens in Y seconds, speed: Z t/s" (some llama.cpp tools)
    if result["tokens_per_second"] is None:
        speed_pattern = r"speed:\s*([\d.]+)\s*t/s"
        match = re.search(speed_pattern, raw_output)
        if match:
            result["tokens_per_second"] = float(match.group(1))

    # Last resort fallback: look for "X tokens per second" in eval time line only
    # IMPORTANT: Only match in lines containing "eval time" to avoid matching prompt eval
    if result["tokens_per_second"] is None:
        # Find lines with eval time (not prompt eval) and extract speed from those
        for line in raw_output.split("\n"):
            if "eval time" in line.lower() and "prompt" not in line.lower():
                simple_match = re.search(r"([\d.]+)\s*tokens per second", line)
                if simple_match:
                    result["tokens_per_second"] = float(simple_match.group(1))
                    break

    return result


def parse_token_counts(raw_output: str) -> dict[str, Optional[int]]:
    """Extract token counts from raw output.

    Returns dict with:
        - prompt_tokens: int or None
        - completion_tokens: int or None
    """
    result = {
        "prompt_tokens": None,
        "completion_tokens": None,
    }

    # Pattern for prompt tokens: "prompt eval time = X ms / Y tokens"
    prompt_pattern = r"prompt eval time\s*=\s*[\d.]+\s*ms\s*/\s*(\d+)\s*tokens"
    match = re.search(prompt_pattern, raw_output)
    if match:
        result["prompt_tokens"] = int(match.group(1))

    # Pattern for completion tokens: "eval time = X ms / Y runs" (NOT prompt eval)
    # Look for the line that has just "eval time" without "prompt" before it
    eval_pattern = r"(?<!prompt\s)eval time\s*=\s*[\d.]+\s*ms\s*/\s*(\d+)\s*(?:runs|tokens)"
    match = re.search(eval_pattern, raw_output)
    if match:
        result["completion_tokens"] = int(match.group(1))

    return result


def parse_acceptance_rate(raw_output: str) -> Optional[float]:
    """Extract speculative decoding acceptance rate.

    Returns acceptance rate as a float (0.0-1.0) or None.
    """
    # Pattern: "accept = X.XXX%" (llama-speculative format)
    # Must handle percentage sign and convert to 0-1 range
    accept_pct_match = re.search(r"accept\s*=\s*([\d.]+)\s*%", raw_output)
    if accept_pct_match:
        rate = float(accept_pct_match.group(1)) / 100
        return rate

    # Pattern: "draft acceptance rate: 0.XXX" or "accept = X.XXX" (without %)
    patterns = [
        r"draft acceptance rate:\s*([\d.]+)",
        r"accept\s*=\s*([\d.]+)",  # Without % sign
        r"acceptance[:\s]+([\d.]+)",
    ]

    for pattern in patterns:
        match = re.search(pattern, raw_output, re.IGNORECASE)
        if match:
            rate = float(match.group(1))
            # Normalize to 0-1 range if given as percentage (>1 means it was a percentage)
            if rate > 1:
                rate = rate / 100
            return rate

    return None


def parse_output(raw_output: str) -> ParsedOutput:
    """Parse all information from raw llama.cpp output.

    Args:
        raw_output: Raw output from llama.cpp.

    Returns:
        ParsedOutput with all extracted information.
    """
    response = parse_response(raw_output)
    timing = parse_timing(raw_output)
    tokens = parse_token_counts(raw_output)
    acceptance = parse_acceptance_rate(raw_output)

    return ParsedOutput(
        response=response,
        tokens_per_second=timing["tokens_per_second"],
        prompt_tokens=tokens["prompt_tokens"],
        completion_tokens=tokens["completion_tokens"],
        total_time_ms=timing["total_time_ms"],
        acceptance_rate=acceptance,
        prompt_eval_time_ms=timing["prompt_eval_time_ms"],
    )


if __name__ == "__main__":
    # Test with standard llama-cli output
    sample_output = """
build: 7404 (52392291b) with GNU 13.3.0 for Linux x86_64
main: llama backend init
llama_model_loader: loaded meta data with 26 key-value pairs
print_info: arch = qwen2
llama_context: n_seq_max = 1
system_info: n_threads = 96

Hello! I'm a helpful AI assistant. How can I help you today?

common_perf_print:    sampling time =      84.91 ms
common_perf_print:        load time =     321.48 ms
common_perf_print: prompt eval time =    1672.75 ms /    62 tokens (   26.98 ms per token,    37.06 tokens per second)
common_perf_print:        eval time =   16012.44 ms /   505 runs   (   31.71 ms per token,    31.54 tokens per second)
common_perf_print:       total time =   17776.75 ms /   567 tokens
    """

    print("=== Output Parser Test: Standard llama-cli ===\n")

    parsed = parse_output(sample_output)

    print(f"Response:\n{parsed.response}\n")
    print(f"Tokens/second: {parsed.tokens_per_second} (expected: 31.54)")
    print(f"Prompt tokens: {parsed.prompt_tokens}")
    print(f"Completion tokens: {parsed.completion_tokens}")
    print(f"Total time: {parsed.total_time_ms} ms")
    print(f"Prompt eval time: {parsed.prompt_eval_time_ms} ms")
    print(f"Acceptance rate: {parsed.acceptance_rate}")

    # Test with llama-speculative output
    spec_output = """
build: 7699 (6b43356a1) with GNU 13.3.0 for Linux x86_64
n_draft   = 16
n_predict = 52
n_drafted = 240
n_accept  = 36
accept    = 15.000%

draft:

llama_perf_context_print:        load time =     625.48 ms
llama_perf_context_print: prompt eval time =    2514.33 ms /    31 tokens (   81.11 ms per token,    12.33 tokens per second)
llama_perf_context_print:        eval time =    3010.31 ms /   225 runs   (   13.38 ms per token,    74.74 tokens per second)
llama_perf_context_print:       total time =    6089.18 ms /   256 tokens

target:

common_perf_print:    sampling time =      13.06 ms
common_perf_print:    samplers time =       5.58 ms /    52 tokens
common_perf_print:        load time =   11265.46 ms
common_perf_print: prompt eval time =    2325.26 ms /   257 tokens (    9.05 ms per token,   110.53 tokens per second)
common_perf_print:        eval time =       0.00 ms /     1 runs   (    0.00 ms per token,      inf tokens per second)
common_perf_print:       total time =    6905.68 ms /   258 tokens
    """

    print("\n=== Output Parser Test: llama-speculative ===\n")

    parsed_spec = parse_output(spec_output)

    # Expected: 52 tokens / 6.906 seconds = 7.53 t/s
    expected_tps = 52 / (6905.68 / 1000)
    print(f"Tokens/second: {parsed_spec.tokens_per_second:.2f} (expected: {expected_tps:.2f})")
    print(f"Total time: {parsed_spec.total_time_ms} ms (expected: 6905.68)")
    print(f"Prompt eval time: {parsed_spec.prompt_eval_time_ms} ms")
    print(f"Acceptance rate: {parsed_spec.acceptance_rate} (expected: 0.15)")

    # Verify correctness
    assert abs(parsed_spec.tokens_per_second - expected_tps) < 0.01, "Speed mismatch!"
    assert abs(parsed_spec.acceptance_rate - 0.15) < 0.001, "Acceptance rate mismatch!"
    print("\n✓ All tests passed!")
