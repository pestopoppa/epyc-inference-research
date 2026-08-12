#!/usr/bin/env python3
"""Render the LFM2.5 GGUF-embedded template for the same five prompts.

LFM2.5's embedded template has NO enable_thinking kwarg: the generation prompt is
unconditionally '<|im_start|>assistant\\n<think>'. There is nothing to toggle, so a
single rendering is the whole story for this arm.
"""
import pathlib

from transformers.utils.chat_template_utils import _compile_jinja_template

OUT = pathlib.Path("/workspace/tmp/wg-lfm-1-thinking")
SRC = pathlib.Path(
    "/workspace/repos/epyc-inference-research/benchmarks/results/scout/"
    "wg-lfm-1-20260812/chat_template_LFM2.5-2.6B-Q4_K_M.jinja"
).read_text()

PROMPTS = [
    "What is the capital city of Japan? Reply with only the city name.",
    "Compute 17 * 23. Reply with only the number.",
    "List the first five prime numbers as a comma-separated list, nothing else.",
    'Return ONLY a JSON object, no prose and no code fence, with keys "name" and "age" '
    "for a person named Ada who is 36 years old.",
    "A shelf holds 3 boxes. Each box holds 4 jars. Each jar holds 6 marbles. "
    "How many marbles in total? Reply with only the number.",
]

tmpl = _compile_jinja_template(SRC)
for i, p in enumerate(PROMPTS):
    s = tmpl.render(
        messages=[{"role": "user", "content": p}],
        bos_token="",
        eos_token="<|im_end|>",
        add_generation_prompt=True,
    )
    (OUT / f"lfm_q{i + 1}.txt").write_text(s)
    (OUT / f"lfm_q{i + 1}.pf").write_text(s + "\n")
    print(f"q{i + 1} chars={len(s)} {s!r}")
