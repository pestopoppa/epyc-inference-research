import json, pathlib, random, sys

# Realistic long prompts built from actual repo content, replacing the synthetic
# filler ("the quick brown fox..." x800) that made the first ngram result an
# upper bound. Real source and prose ARE somewhat self-similar — that is genuine,
# and it is what these roles actually serve — but they are nothing like one
# sentence repeated 800 times.
SOURCES = [
    "/mnt/raid0/llm/epyc-orchestrator/scripts/server/orchestrator_stack.py",
    "/mnt/raid0/llm/epyc-orchestrator/scripts/server/stack_numa.py",
    "/mnt/raid0/llm/epyc-inference-research/scripts/benchmark/server_numa_np_sweep.py",
    "/mnt/raid0/llm/epyc-inference-research/docs/protocols/model-registration-runbook.md",
    "/workspace/handoffs/active/numa-placement-defect-20260730.md",
]

chunks = []
for s in SOURCES:
    p = pathlib.Path(s)
    if p.exists():
        chunks.append(p.read_text(errors="replace"))
if not chunks:
    sys.exit("no source files found")

corpus = "\n\n".join(chunks)
words = corpus.split()
print(f"corpus: {len(words):,} words from {len(chunks)} files")

OUT = pathlib.Path("/mnt/raid0/llm/tmp")
for name, approx_tok in [("r8k", 8000), ("r32k", 32000)]:
    # ~0.75 words per token for code/markdown (punctuation splits into tokens)
    need = int(approx_tok * 0.72)
    if need > len(words):
        need = len(words)
        print(f"  WARNING {name}: corpus only {len(words)} words, short of target")
    text = " ".join(words[:need])
    body = {
        "messages": [{"role": "user", "content":
            "Here is an excerpt from our codebase and documentation:\n\n" + text +
            "\n\nSummarize what this code and documentation are for, in three sentences."}],
        "max_tokens": 256, "temperature": 0.3, "seed": 42,
    }
    f = OUT / f"req_{name}.json"
    f.write_text(json.dumps(body))
    print(f"  {name}: {need:,} words -> {f.stat().st_size/1024:.0f} KB")
