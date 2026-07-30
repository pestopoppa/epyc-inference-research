import json, pathlib

# Filler that tokenizes near ~1 token per word for an English-ish stream.
WORD = "the quick brown fox jumps over a lazy dog while counting numbers "
OUT = pathlib.Path("/mnt/raid0/llm/tmp")

for name, approx_tok in [("p0k5", 500), ("p8k", 8000), ("p32k", 32000)]:
    # ~11 tokens per WORD repetition; overshoot slightly then trim by words.
    reps = max(1, approx_tok // 11)
    filler = (WORD * reps).strip()
    body = {
        "messages": [{
            "role": "user",
            "content": (
                "Here is a log excerpt:\n\n" + filler +
                "\n\nSummarize the excerpt in exactly three sentences."
            ),
        }],
        "max_tokens": 256,
        "temperature": 0.3,
        "seed": 42,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    p = OUT / f"req_{name}.json"
    p.write_text(json.dumps(body))
    print(f"{name}: ~{approx_tok} tok target, {len(filler.split())} words, {p.stat().st_size} bytes")
