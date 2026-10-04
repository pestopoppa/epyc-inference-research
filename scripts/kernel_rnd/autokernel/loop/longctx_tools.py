#!/usr/bin/env python3
"""Offline tooling for the long-context surface (longctx.py). No server, no inference.

    # one frozen long prompt per target, from a real corpus already on disk
    PYTHONPATH=.:scripts/kernel_rnd python3 -m scripts.kernel_rnd.autokernel.loop.longctx_tools build-manifest \\
        --recipe <target>.launch.json --target-id <target id> \\
        --tokenizer <HF tokenizer.json matching the GGUF vocab> --format chatml|raw \\
        --corpus /mnt/raid0/llm/data/wikitext2_test.txt --out-dir <campaign>/inputs \\
        --name <stem> [--depth 65536] [--tail 4096] [--production-log <llama-server-N.log>]

    # the production context histogram the planner is fed (the loop regenerates it too)
    PYTHONPATH=.:scripts/kernel_rnd python3 -m scripts.kernel_rnd.autokernel.loop.longctx_tools histogram \\
        /mnt/raid0/llm/epyc-orchestrator/logs/llama-server-*.log [--out file.json]

The tokenizer is the model's HF `tokenizer.json`; `--check-manifest` round-trips the
target's EXISTING short manifest through it (decode then encode must reproduce its token
IDs exactly) so a vocabulary that differs from the GGUF is refused before anything is
written. The first live slot generation cross-checks again: the server must report the
prefix length as evaluated prompt tokens.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

from .. import schemas
from . import longctx, planned_serving, serving

#: Literal turn templates with special tokens inline; the tokenizer parses them as added
#: tokens (the round-trip check proves it for the target's own short manifest).
FORMATS = {
    "chatml": ("<|im_start|>user\nRead the following document carefully; a question about it "
               "follows at the end.\n\n",
               "\n\nSummarize the document above in three paragraphs.<|im_end|>\n"
               "<|im_start|>assistant\n<think>\n\n</think>\n\n"),
    "raw": ("<｜begin▁of▁sentence｜>Read the following document carefully.\n\n",
            "\n\nA three-paragraph summary of the document above:\n"),
}
SLACK_TOKENS = 16
IDENTITY_TOKENS = 32


def _file_sha(path: Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def build(*, template: serving.Recipe, target_id: str, encode, corpus_text: str,
          fmt: str, depth: int, tail: int, version: str, manifest_path: Path,
          production_logs: list[str], source: dict) -> tuple[dict, dict]:
    """(frozen prompt manifest, long-context spec) for one target. Pure given `encode`."""
    if fmt not in FORMATS:
        raise SystemExit(f"unknown format {fmt!r}; one of {sorted(FORMATS)}")
    if template.np != 1:
        raise SystemExit("the long-context surface serves one slot (np=1)")
    header, closer = (encode(text) for text in FORMATS[fmt])
    document = encode(corpus_text)
    body = depth - len(header)
    if body <= 0 or len(document) < body + tail:
        raise SystemExit(f"corpus too short: {len(document)} tokens for a {depth}-token "
                         f"prefix plus a {tail}-token tail")
    prefix = header + document[:body]
    tail_ids = document[body:body + tail]
    request = {"prompt": prefix + closer, "n_predict": template.n_predict,
               "temperature": template.temperature, "top_p": template.top_p,
               "top_k": template.top_k, "cache_prompt": True, "seed": 42,
               "ignore_eos": True, "return_tokens": True, "stream": False}
    raw = planned_serving._canonical(request)
    prompt = planned_serving.FrozenPrompt.from_dict(
        {"prompt_id": f"{version}-p01", "request": request,
         "request_digest": hashlib.sha256(raw).hexdigest()},
        schema=planned_serving.PROMPT_SCHEMA_V2)
    manifest = {"schema": planned_serving.PROMPT_SCHEMA_V2, "version": version,
                "prompts": [prompt.to_dict()]}
    manifest["digest"] = schemas.content_hash(
        {k: manifest[k] for k in ("schema", "version", "prompts")})
    loaded = planned_serving.FrozenPromptManifest.from_dict(manifest)
    loaded.requests((prompt.prompt_id,), template)   # workload must match the recipe
    occupied = depth + max(tail + 1, len(closer) + max(template.n_predict, IDENTITY_TOKENS))
    spec = {"schema": longctx.SPEC_SCHEMA, "target_id": target_id, "version": version,
            "prompt_manifest": str(Path(manifest_path).resolve()),
            "prompt_manifest_digest": loaded.digest, "prefix_tokens": depth,
            "tail": tail_ids, "ctx": int(math.ceil((occupied + 512) / 1024) * 1024),
            "identity_tokens": IDENTITY_TOKENS,
            "bounds": {"a_prompt_n_max": tail + SLACK_TOKENS,
                       "b_prompt_n_max": len(closer) + SLACK_TOKENS},
            "production_logs": list(production_logs),
            "source": {**source, "format": fmt, "document_tokens": len(document),
                       "document_tokens_used": [0, body + tail]}}
    spec["digest"] = longctx.spec_digest(spec)
    return manifest, spec


def _check_tokenizer(tokenizer, manifest_path: Path) -> None:
    body = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    for item in body["prompts"]:
        ids = item["request"]["prompt"]
        if not isinstance(ids, list):
            continue
        again = tokenizer.encode(tokenizer.decode(ids, skip_special_tokens=False),
                                 add_special_tokens=False).ids
        if again != ids:
            raise SystemExit(f"tokenizer does not round-trip {manifest_path}: its vocabulary "
                             "differs from the one that built the target's manifest")


def _build_manifest(args) -> int:
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    if args.check_manifest is not None:
        _check_tokenizer(tokenizer, args.check_manifest)
    document = json.loads(Path(args.recipe).read_text(encoding="utf-8"))
    # A canonical launch JSON carries the exact template the loop serves; a recipe JSON is it.
    template = serving.Recipe.from_dict(document.get("template", document))
    out_dir = Path(args.out_dir)
    manifest_path = out_dir / f"{args.name}.prompt-manifest.json"
    spec_path = out_dir / f"{args.name}.longctx.json"
    for path in (manifest_path, spec_path):
        if path.exists() and not args.force:
            raise SystemExit(f"{path} exists; refusing to overwrite (pass --force)")
    corpus_text = Path(args.corpus).read_text(encoding="utf-8")
    manifest, spec = build(
        template=template, target_id=args.target_id,
        encode=lambda text: tokenizer.encode(text, add_special_tokens=False).ids,
        corpus_text=corpus_text, fmt=args.format, depth=args.depth, tail=args.tail,
        version=args.name, manifest_path=manifest_path,
        production_logs=[str(Path(p).resolve()) for p in args.production_log],
        source={"corpus": str(Path(args.corpus).resolve()),
                "corpus_sha256": _file_sha(args.corpus),
                "tokenizer": str(Path(args.tokenizer).resolve()),
                "tokenizer_sha256": _file_sha(args.tokenizer),
                "tokenizer_checked_against": (None if args.check_manifest is None
                                              else str(Path(args.check_manifest).resolve()))})
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, separators=(",", ":")) + "\n",
                             encoding="utf-8")
    spec_path.write_text(json.dumps(spec, separators=(",", ":")) + "\n", encoding="utf-8")
    loaded = longctx.Spec.load(spec_path)          # the loop's own validator
    print(json.dumps({"spec": str(spec_path), "manifest": str(manifest_path),
                      "depth": loaded.depth, "tail": len(loaded.tail),
                      "probe": len(loaded.probe), "ctx": loaded.body["ctx"],
                      "digest": loaded.digest}, indent=1))
    return 0


def _histogram(args) -> int:
    body = longctx.parse_server_logs(args.logs)
    text = json.dumps(body, indent=1, sort_keys=True) + "\n"
    if args.out is not None:
        Path(args.out).write_text(text, encoding="utf-8")
    sys.stdout.write(text)
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build-manifest")
    b.add_argument("--recipe", type=Path, required=True,
                   help="the target's canonical launch JSON (its template) or recipe JSON")
    b.add_argument("--target-id", required=True)
    b.add_argument("--tokenizer", type=Path, required=True)
    b.add_argument("--format", choices=sorted(FORMATS), required=True)
    b.add_argument("--corpus", type=Path, required=True)
    b.add_argument("--out-dir", type=Path, required=True)
    b.add_argument("--name", required=True)
    b.add_argument("--depth", type=int, default=65536)
    b.add_argument("--tail", type=int, default=4096)
    b.add_argument("--production-log", action="append", default=[])
    b.add_argument("--check-manifest", type=Path,
                   help="the target's existing frozen prompt manifest to round-trip")
    b.add_argument("--force", action="store_true")
    h = sub.add_parser("histogram")
    h.add_argument("logs", nargs="+", type=Path)
    h.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    return _build_manifest(args) if args.command == "build-manifest" else _histogram(args)


if __name__ == "__main__":
    raise SystemExit(main())
