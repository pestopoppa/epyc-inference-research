#!/usr/bin/env python3
"""Dump selected GGUF tensor and metadata descriptors without loading a model.

This is intended for kernel-prep audits such as GLM NextN/MTP tail-block mapping.
It uses gguf-py's header/tensor descriptor reader and does not start inference.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence


DEFAULT_GGUF_PY = Path("/mnt/raid0/llm/llama.cpp-experimental/gguf-py")


@dataclass(frozen=True)
class TensorDescriptor:
    file: str
    name: str
    layer: int | None
    shape: list[int]
    n_elements: int
    n_bytes: int
    tensor_type: str
    data_offset: int


def import_gguf_reader(gguf_py: Path) -> Any:
    sys.path.insert(0, str(gguf_py))
    try:
        from gguf import GGUFReader  # type: ignore
    except ModuleNotFoundError as exc:
        if exc.name == "numpy":
            raise SystemExit(
                "NumPy is required by gguf-py. Run with a Python environment that has numpy, "
                "or use: uv run --with numpy python scripts/benchmark/gguf_tensor_contract.py ..."
            ) from exc
        raise
    return GGUFReader


def layer_index(name: str) -> int | None:
    match = re.search(r"(?:^|\.)blk\.(\d+)\.", name)
    if not match:
        return None
    return int(match.group(1))


def compile_regexes(patterns: Sequence[str]) -> list[re.Pattern[str]]:
    return [re.compile(pattern) for pattern in patterns]


def matches_any(patterns: Sequence[re.Pattern[str]], value: str) -> bool:
    return not patterns or any(pattern.search(value) for pattern in patterns)


def layer_allowed(layer: int | None, layer_start: int | None, layer_end: int | None) -> bool:
    if layer_start is None and layer_end is None:
        return True
    if layer is None:
        return False
    if layer_start is not None and layer < layer_start:
        return False
    if layer_end is not None and layer >= layer_end:
        return False
    return True


def discover_gguf_files(paths: Sequence[Path]) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        if path.is_dir():
            files.extend(sorted(path.rglob("*.gguf")))
        elif path.is_file():
            files.append(path)
        else:
            raise SystemExit(f"GGUF path does not exist: {path}")
    unique: dict[str, Path] = {}
    for file in files:
        unique[str(file.resolve())] = file
    return [unique[key] for key in sorted(unique)]


def field_value(field: Any) -> Any:
    try:
        value = field.contents()
    except Exception:  # pragma: no cover - defensive against gguf API drift
        value = field.parts[field.data[0]].tolist()
    return value


def read_contract(
    files: Sequence[Path],
    *,
    gguf_py: Path,
    tensor_patterns: Sequence[str],
    metadata_patterns: Sequence[str],
    layer_start: int | None,
    layer_end: int | None,
) -> dict[str, Any]:
    GGUFReader = import_gguf_reader(gguf_py)
    tensor_regexes = compile_regexes(tensor_patterns)
    metadata_regexes = compile_regexes(metadata_patterns)

    file_entries: list[dict[str, Any]] = []
    all_matches: list[TensorDescriptor] = []
    metadata_matches: dict[str, dict[str, Any]] = {}
    total_tensors = 0

    for file in files:
        reader = GGUFReader(str(file), "r")
        file_tensor_matches: list[TensorDescriptor] = []
        total_tensors += len(reader.tensors)

        for key, field in reader.fields.items():
            if matches_any(metadata_regexes, key):
                metadata_matches.setdefault(str(file), {})[key] = field_value(field)

        for tensor in reader.tensors:
            name = str(tensor.name)
            layer = layer_index(name)
            if not layer_allowed(layer, layer_start, layer_end):
                continue
            if not matches_any(tensor_regexes, name):
                continue
            desc = TensorDescriptor(
                file=str(file),
                name=name,
                layer=layer,
                shape=[int(dim) for dim in tensor.shape.tolist()],
                n_elements=int(tensor.n_elements),
                n_bytes=int(tensor.n_bytes),
                tensor_type=str(tensor.tensor_type.name),
                data_offset=int(tensor.data_offset),
            )
            file_tensor_matches.append(desc)
            all_matches.append(desc)

        file_entries.append({
            "file": str(file),
            "tensor_count": len(reader.tensors),
            "matched_tensor_count": len(file_tensor_matches),
        })

    return {
        "files": file_entries,
        "summary": {
            "file_count": len(files),
            "total_tensors": total_tensors,
            "matched_tensors": len(all_matches),
            "layer_start": layer_start,
            "layer_end": layer_end,
            "tensor_regexes": list(tensor_patterns),
            "metadata_regexes": list(metadata_patterns),
        },
        "metadata": metadata_matches,
        "tensors": [asdict(desc) for desc in all_matches],
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path, help="GGUF file(s) or directories")
    parser.add_argument("--gguf-py", type=Path, default=DEFAULT_GGUF_PY)
    parser.add_argument(
        "--tensor-regex",
        action="append",
        default=[],
        help="Tensor-name regex. May be repeated. Empty means all tensors in the layer range.",
    )
    parser.add_argument(
        "--metadata-regex",
        action="append",
        default=[r"general\.architecture", r"(nextn|block_count|attention\.indexer)"],
        help="Metadata-key regex. May be repeated.",
    )
    parser.add_argument("--layer-start", type=int, default=None, help="Inclusive layer lower bound")
    parser.add_argument("--layer-end", type=int, default=None, help="Exclusive layer upper bound")
    parser.add_argument("--output", type=Path, default=None, help="Write JSON here instead of stdout")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    files = discover_gguf_files(args.paths)
    if not files:
        raise SystemExit("no GGUF files found")
    contract = read_contract(
        files,
        gguf_py=args.gguf_py,
        tensor_patterns=args.tensor_regex,
        metadata_patterns=args.metadata_regex,
        layer_start=args.layer_start,
        layer_end=args.layer_end,
    )
    payload = json.dumps(contract, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)
    else:
        print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
