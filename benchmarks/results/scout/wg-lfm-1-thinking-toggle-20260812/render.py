#!/usr/bin/env python3
"""Render the GGUF-embedded gemma4 chat template for the five WG-LFM-1 prompts,
once with enable_thinking=True and once with enable_thinking=False.

The rendered strings are what llama.cpp itself would build (common/chat.cpp
common_chat_template_direct_apply_impl passes exactly messages / bos_token /
eos_token / enable_thinking into the jinja context, plus add_generation_prompt).

bos_token is rendered EMPTY on purpose: the GGUF sets tokenizer.ggml.add_bos_token
= True, so llama-completion's common_tokenize(..., add_special=true, parse_special=true)
prepends <bos> itself. Rendering it here as well would double it.
"""
import hashlib
import json
import pathlib

import jinja2

OUT = pathlib.Path("/workspace/tmp/wg-lfm-1-thinking")
TEMPLATE = (OUT / "gemma_chat_template.jinja").read_text()

PROMPTS = [
    "What is the capital city of Japan? Reply with only the city name.",
    "Compute 17 * 23. Reply with only the number.",
    "List the first five prime numbers as a comma-separated list, nothing else.",
    'Return ONLY a JSON object, no prose and no code fence, with keys "name" and "age" '
    "for a person named Ada who is 36 years old.",
    "A shelf holds 3 boxes. Each box holds 4 jars. Each jar holds 6 marbles. "
    "How many marbles in total? Reply with only the number.",
]

env = jinja2.Environment(
    trim_blocks=True, lstrip_blocks=True, keep_trailing_newline=True,
    undefined=jinja2.Undefined,  # llama.cpp/minja omits `tools` entirely when there are none
)
env.policies["json.dumps_kwargs"] = {"ensure_ascii": False}
tmpl = env.from_string(TEMPLATE)

manifest = []
for i, p in enumerate(PROMPTS):
    for flag, tag in ((True, "on"), (False, "off")):
        s = tmpl.render(
            messages=[{"role": "user", "content": p}],
            bos_token="",
            eos_token="<eos>",
            add_generation_prompt=True,
            enable_thinking=flag,
        )
        f = OUT / f"prompt_q{i + 1}_think{tag}.txt"
        f.write_text(s)
        manifest.append({
            "q": i + 1, "think": tag, "file": str(f),
            "chars": len(s), "sha256": hashlib.sha256(s.encode()).hexdigest(),
            "repr": repr(s),
        })

(OUT / "render_manifest.json").write_text(json.dumps(manifest, indent=2))
for m in manifest:
    print(f"q{m['q']} think={m['think']:3s} chars={m['chars']:4d} sha={m['sha256'][:16]} {m['repr']}")
