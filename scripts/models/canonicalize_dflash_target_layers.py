#!/usr/bin/env python3
"""Copy a legacy DFlash GGUF into the current upstream metadata/tensor grammar.

The source is never modified. Tensor payloads, tensor types, and tensor ordering
are copied verbatim through gguf-py's reader/writer. Only the legacy target-layer
key and the two pre-upstream tensor names are canonicalized. The command fails
closed if the expected legacy grammar is incomplete or ambiguous.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


DEFAULT_GGUF_PY = Path("/mnt/raid0/llm/llama.cpp/gguf-py")
LEGACY_KEY = "dflash.target_layer_ids"
CANONICAL_KEY = "dflash.target_layers"
TENSOR_RENAMES = {
    "dflash.fc.weight": "fc.weight",
    "dflash.hidden_norm.weight": "enc.output_norm.weight",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--gguf-python-root", type=Path, default=DEFAULT_GGUF_PY)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.input.resolve() == args.output.resolve():
        raise RuntimeError("input and output must differ; the source is immutable")
    if not args.input.is_file():
        raise RuntimeError(f"input GGUF is absent: {args.input}")
    if args.output.exists():
        raise RuntimeError(f"output already exists: {args.output}")
    if not args.gguf_python_root.is_dir():
        raise RuntimeError(f"gguf-py root is absent: {args.gguf_python_root}")

    sys.path.insert(0, str(args.gguf_python_root))
    import gguf  # noqa: PLC0415

    reader = gguf.GGUFReader(args.input, "r")
    if CANONICAL_KEY in reader.fields:
        raise RuntimeError(f"source already contains {CANONICAL_KEY}")
    legacy = reader.get_field(LEGACY_KEY)
    if legacy is None:
        raise RuntimeError(f"source lacks required legacy key {LEGACY_KEY}")
    target_layers = legacy.contents()
    if not isinstance(target_layers, list) or not target_layers or any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in target_layers
    ):
        raise RuntimeError(f"invalid legacy target layer array: {target_layers!r}")

    architecture = reader.get_field(gguf.Keys.General.ARCHITECTURE)
    if architecture is None or architecture.contents() != "dflash":
        raise RuntimeError("source general.architecture is not dflash")
    writer = gguf.GGUFWriter(args.output, arch="dflash", endianess=reader.endianess)
    alignment = reader.get_field(gguf.Keys.General.ALIGNMENT)
    if alignment is not None:
        writer.data_alignment = alignment.contents()

    for field in reader.fields.values():
        if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith("GGUF."):
            continue
        value_type = field.types[0]
        sub_type = field.types[-1] if value_type == gguf.GGUFValueType.ARRAY else None
        writer.add_key_value(field.name, field.contents(), value_type, sub_type=sub_type)
    writer.add_key_value(
        CANONICAL_KEY,
        target_layers,
        gguf.GGUFValueType.ARRAY,
        sub_type=gguf.GGUFValueType.INT32,
    )

    source_tensor_names = {tensor.name for tensor in reader.tensors}
    missing_legacy_tensors = sorted(set(TENSOR_RENAMES) - source_tensor_names)
    canonical_collisions = sorted(set(TENSOR_RENAMES.values()) & source_tensor_names)
    if missing_legacy_tensors or canonical_collisions:
        raise RuntimeError(
            "legacy tensor grammar is incomplete or ambiguous: "
            f"missing={missing_legacy_tensors}, collisions={canonical_collisions}"
        )

    for tensor in reader.tensors:
        writer.add_tensor_info(
            TENSOR_RENAMES.get(tensor.name, tensor.name),
            tensor.data.shape,
            tensor.data.dtype,
            tensor.data.nbytes,
            tensor.tensor_type,
        )
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    for tensor in reader.tensors:
        writer.write_tensor_data(tensor.data, tensor_endianess=reader.endianess)
    writer.close()

    if not args.output.is_file() or args.output.stat().st_size <= 0:
        raise RuntimeError("canonicalized output was not created")
    print(f"created {args.output} ({args.output.stat().st_size} bytes)")
    print(f"added {CANONICAL_KEY}={target_layers}")
    print(f"renamed tensors: {TENSOR_RENAMES}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
