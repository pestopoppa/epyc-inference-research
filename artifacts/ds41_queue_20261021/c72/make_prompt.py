"""DS41-C7.2 prompt builder: an EXACT-N-token natural-text prompt as a JSON array of token ids (host side, no inference).

usage: make_prompt.py <tokenizer.json> <corpus_dir> <n_tokens> <out.json>

Why exact: the node profiler accumulates every graph eval after GGML_CPU_PROF_SKIP, and SKIP must be set at launch.
With a known prompt length the prefill graph count is ceil(n_tokens / n_ubatch), so SKIP is exact instead of guessed.
The prompt is BOS + the first (n_tokens-1) tokens of the sorted *.md corpus, so it is natural text (the indexer sees
realistic keys) and byte-identical across runs for the same corpus commit. The model sees ids, not text.
"""
import sys, json, hashlib, pathlib
from tokenizers import Tokenizer

tok_path, corpus_dir, n, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
tk = Tokenizer.from_file(tok_path)
bos = tk.token_to_id("<｜begin▁of▁sentence｜>")
assert bos is not None, "BOS token not in tokenizer.json"
files = sorted(pathlib.Path(corpus_dir).rglob("*.md"))
ids, used = [bos], []
for f in files:
    try:
        text = f.read_text(encoding="utf-8")
    except Exception:
        continue
    ids.extend(tk.encode(text, add_special_tokens=False).ids)
    used.append(str(f))
    if len(ids) >= n:
        break
if len(ids) < n:
    sys.exit(f"corpus too small: {len(ids)} < {n} tokens")
ids = ids[:n]
meta = {"n_tokens": n, "bos_id": bos, "ids_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
        "tokenizer": tok_path, "corpus_dir": corpus_dir, "n_files_used": len(used), "last_file": used[-1]}
json.dump({"meta": meta, "ids": ids}, open(out, "w"))
print(json.dumps(meta))
