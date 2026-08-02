#!/usr/bin/env python3
"""One vision query against the Qwen3-VL server (port 8802)."""
import base64
import json
import sys
import time
import urllib.request

IMG = sys.argv[1] if len(sys.argv) > 1 else "/mnt/raid0/llm/epyc-inference-research/test_images/vl_rubric/chart_bar.png"
PROMPT = sys.argv[2] if len(sys.argv) > 2 else "Describe this image in detail, including any text, numbers and structure you can see."

with open(IMG, "rb") as f:
    b64 = base64.b64encode(f.read()).decode()

body = json.dumps({
    "messages": [{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            {"type": "text", "text": PROMPT},
        ],
    }],
    "max_tokens": 160,
    "temperature": 0.7,
    "seed": 42,
}).encode()

req = urllib.request.Request(
    "http://127.0.0.1:8802/v1/chat/completions",
    data=body, headers={"Content-Type": "application/json"},
)
t0 = time.time()
with urllib.request.urlopen(req, timeout=600) as r:
    out = json.loads(r.read())
wall = time.time() - t0
txt = out["choices"][0]["message"]["content"]
print(json.dumps({
    "wall_s": round(wall, 3),
    "completion_tokens": out.get("usage", {}).get("completion_tokens"),
    "prompt_tokens": out.get("usage", {}).get("prompt_tokens"),
    "text_head": txt[:180].replace("\n", " "),
}))
