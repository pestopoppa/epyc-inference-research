#!/usr/bin/env python3
"""Dump identity metadata + SHA-256 of the GGUF-EMBEDDED chat template.

WG-LFM-1 requires the embedded template (not the LEAP sidecars) to be hashed
and archived, because the sidecars omit the reasoning prefill / tool rendering.
"""
import hashlib
import json
import sys

sys.path.insert(0, "/mnt/raid0/llm/llama.cpp/gguf-py")
from gguf import GGUFReader  # noqa: E402


def val(reader, key):
    f = reader.fields.get(key)
    if f is None:
        return None
    try:
        return f.contents()
    except Exception:
        return None


def main(path):
    r = GGUFReader(path)
    tmpl = val(r, "tokenizer.chat_template")
    out = {
        "file": path,
        "architecture": val(r, "general.architecture"),
        "name": val(r, "general.name"),
        "file_type": val(r, "general.file_type"),
        "quantization_version": val(r, "general.quantization_version"),
        "context_length": val(r, f"{val(r, 'general.architecture')}.context_length"),
        "block_count": val(r, f"{val(r, 'general.architecture')}.block_count"),
        "embedding_length": val(r, f"{val(r, 'general.architecture')}.embedding_length"),
        "n_tensors": len(r.tensors),
        "chat_template_present": tmpl is not None,
        "chat_template_len": len(tmpl) if tmpl else 0,
        "chat_template_sha256": hashlib.sha256(tmpl.encode("utf-8")).hexdigest()
        if tmpl
        else None,
    }
    print(json.dumps(out, indent=2, default=str))
    if tmpl:
        base = path.rsplit("/", 1)[-1].replace(".gguf", "")
        p = f"/workspace/tmp/wg-lfm-1/chat_template_{base}.jinja"
        with open(p, "w") as fh:
            fh.write(tmpl)
        print(f"# template archived -> {p}", file=sys.stderr)


if __name__ == "__main__":
    for p in sys.argv[1:]:
        main(p)
