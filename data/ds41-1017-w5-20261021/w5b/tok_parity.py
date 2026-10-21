"""DS41-W5b static check: llama.cpp (vocab-only llama-tokenize) vs HF tokenizer.json token ids for the
W5 coherence prompts (both the failing article+question+Answer: family and the passing ones).

usage: tok_parity.py <prompts.jsonl> [<prompts.jsonl> ...] -- out.json
"""
import json, os, subprocess, sys, tempfile
from tokenizers import Tokenizer

B = "/mnt/raid0/llm/llama.cpp-experimental-ds41-20261005/build-cpu/bin"
M = "/mnt/raid0/llm/models/antirez/deepseek-v4.1-flash-gguf/DeepSeek-V4.1-Flash-1017-MXFP4-experts-00001-of-00009.gguf"
W = "/mnt/raid0/llm/tmp/ds41-w5ab-w62-20261021/w5b"
args = sys.argv[1:]
out = args[args.index("--") + 1]
files = args[:args.index("--")]
tok = Tokenizer.from_file(f"{W}/tokenizer.json")
rows = []
for f in files:
    for line in open(f):
        p = json.loads(line)
        with tempfile.NamedTemporaryFile("w", dir=W, suffix=".txt", delete=False) as fh:
            fh.write(p["prompt"])
            tmp = fh.name
        try:
            r = subprocess.run([f"{B}/llama-tokenize", "-m", M, "-f", tmp, "--ids", "--log-disable"],
                               capture_output=True, text=True, env=dict(os.environ, LD_LIBRARY_PATH=B), timeout=300)
        finally:
            os.unlink(tmp)
        ids_cpp = json.loads(r.stdout.strip().splitlines()[-1])
        ids_hf = tok.encode(p["prompt"], add_special_tokens=False).ids
        body_cpp = ids_cpp[1:] if ids_cpp and ids_cpp[0] == 0 else ids_cpp
        first = next((i for i, (a, b) in enumerate(zip(body_cpp, ids_hf)) if a != b), None)
        if first is None and len(body_cpp) != len(ids_hf):
            first = min(len(body_cpp), len(ids_hf))
        rows.append({"id": p["id"], "n_cpp": len(ids_cpp), "cpp_bos": bool(ids_cpp and ids_cpp[0] == 0),
                     "n_hf": len(ids_hf), "identical_after_bos": first is None, "first_diff": first,
                     "tail_cpp": [tok.id_to_token(i) for i in body_cpp[-8:]]})
        print(json.dumps(rows[-1]), flush=True)
json.dump(rows, open(out, "w"), indent=1)
