#!/usr/bin/env python3
"""Render the LFM2.5-1.2B-Instruct GGUF-embedded template for the five WG-LFM-1 prompts.

The -Instruct template has NO `enable_thinking` kwarg and NO unconditional `<think>`
prefill: for a single user message with no system prompt and no tools, every branch
except the message loop and `add_generation_prompt` is inert, so the render reduces to

    {bos}<|im_start|>user\\n{content}<|im_end|>\\n<|im_start|>assistant\\n

This hand-reduction is NOT trusted on its own — check_render.sh diffs the token
sequence produced from these files against llama.cpp's own minja render of the same
GGUF-embedded template (`llama-cli --jinja -st`), which is the control.

bos_token rendered EMPTY on purpose: llama-completion's common_tokenize(add_special=true)
prepends BOS itself when tokenizer.ggml.add_bos_token is set.
"""
import hashlib
import json
import pathlib

OUT = pathlib.Path("/workspace/tmp/wg-lfmi")

PROMPTS = [
    "What is the capital city of Japan? Reply with only the city name.",
    "Compute 17 * 23. Reply with only the number.",
    "List the first five prime numbers as a comma-separated list, nothing else.",
    'Return ONLY a JSON object, no prose and no code fence, with keys "name" and "age" '
    "for a person named Ada who is 36 years old.",
    "A shelf holds 3 boxes. Each box holds 4 jars. Each jar holds 6 marbles. "
    "How many marbles in total? Reply with only the number.",
]

manifest = []
for i, p in enumerate(PROMPTS):
    s = f"<|im_start|>user\n{p}<|im_end|>\n<|im_start|>assistant\n"
    (OUT / f"lfmi_q{i + 1}.txt").write_text(s)
    # trailing \n: llama.cpp's -f loader strips exactly one trailing newline
    (OUT / f"lfmi_q{i + 1}.pf").write_text(s + "\n")
    (OUT / f"raw_q{i + 1}.txt").write_text(p)
    manifest.append({
        "q": i + 1, "chars": len(s), "prompt": p,
        "sha256": hashlib.sha256(s.encode()).hexdigest(), "repr": repr(s),
        "has_think_prefill": "<think>" in s,
    })

(OUT / "render_manifest.json").write_text(json.dumps(manifest, indent=2))
for m in manifest:
    print(f"q{m['q']} chars={m['chars']:4d} think_prefill={m['has_think_prefill']} {m['repr']}")
