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
TARGET_HIDDEN_SIZE_KEY = "dflash.target_hidden_size"
LEGACY_MASK_KEY = "dflash.mask_token_id"
CANONICAL_MASK_KEY = "tokenizer.ggml.mask_token_id"
TENSOR_RENAMES = {
    "dflash.fc.weight": "fc.weight",
    "dflash.hidden_norm.weight": "enc.output_norm.weight",
}
OMIT_TARGET_OWNED_TENSORS = {"token_embd.weight", "output.weight"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--target-model", type=Path, required=True)
    parser.add_argument("--gguf-python-root", type=Path, default=DEFAULT_GGUF_PY)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.input.resolve() == args.output.resolve():
        raise RuntimeError("input and output must differ; the source is immutable")
    if not args.input.is_file():
        raise RuntimeError(f"input GGUF is absent: {args.input}")
    if not args.target_model.is_file():
        raise RuntimeError(f"target GGUF is absent: {args.target_model}")
    if args.output.exists():
        raise RuntimeError(f"output already exists: {args.output}")
    if not args.gguf_python_root.is_dir():
        raise RuntimeError(f"gguf-py root is absent: {args.gguf_python_root}")

    sys.path.insert(0, str(args.gguf_python_root))
    import gguf  # noqa: PLC0415

    reader = gguf.GGUFReader(args.input, "r")
    target_reader = gguf.GGUFReader(args.target_model, "r")
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
    target_architecture = target_reader.get_field(gguf.Keys.General.ARCHITECTURE)
    target_architecture_name = target_architecture.contents() if target_architecture else None
    if not isinstance(target_architecture_name, str) or target_architecture_name == "dflash":
        raise RuntimeError(f"invalid target architecture: {target_architecture_name!r}")
    target_hidden_field = target_reader.get_field(f"{target_architecture_name}.embedding_length")
    target_hidden_size = target_hidden_field.contents() if target_hidden_field else None
    if isinstance(target_hidden_size, bool) or not isinstance(target_hidden_size, int) or target_hidden_size <= 0:
        raise RuntimeError(f"invalid target embedding length: {target_hidden_size!r}")
    legacy_mask_field = reader.get_field(LEGACY_MASK_KEY)
    mask_token_id = legacy_mask_field.contents() if legacy_mask_field else None
    if isinstance(mask_token_id, bool) or not isinstance(mask_token_id, int) or mask_token_id < 0:
        raise RuntimeError(f"invalid legacy mask token id: {mask_token_id!r}")
    target_tokenizer_fields = [
        field for field in target_reader.fields.values() if field.name.startswith("tokenizer.")
    ]
    if not target_tokenizer_fields or target_reader.get_field(gguf.Keys.Tokenizer.PRE) is None:
        raise RuntimeError("target GGUF lacks canonical tokenizer metadata")
    writer = gguf.GGUFWriter(args.output, arch="dflash", endianess=reader.endianess)
    alignment = reader.get_field(gguf.Keys.General.ALIGNMENT)
    if alignment is not None:
        writer.data_alignment = alignment.contents()

    for field in reader.fields.values():
        if (
            field.name == gguf.Keys.General.ARCHITECTURE
            or field.name.startswith("GGUF.")
            or field.name.startswith("tokenizer.")
        ):
            continue
        value_type = field.types[0]
        sub_type = field.types[-1] if value_type == gguf.GGUFValueType.ARRAY else None
        writer.add_key_value(field.name, field.contents(), value_type, sub_type=sub_type)
    for field in target_tokenizer_fields:
        if field.name == CANONICAL_MASK_KEY:
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
    writer.add_key_value(TARGET_HIDDEN_SIZE_KEY, target_hidden_size, gguf.GGUFValueType.UINT32)
    writer.add_key_value(CANONICAL_MASK_KEY, mask_token_id, gguf.GGUFValueType.UINT32)

    source_tensor_names = {tensor.name for tensor in reader.tensors}
    missing_legacy_tensors = sorted(set(TENSOR_RENAMES) - source_tensor_names)
    canonical_collisions = sorted(set(TENSOR_RENAMES.values()) & source_tensor_names)
    if missing_legacy_tensors or canonical_collisions:
        raise RuntimeError(
            "legacy tensor grammar is incomplete or ambiguous: "
            f"missing={missing_legacy_tensors}, collisions={canonical_collisions}"
        )
    missing_target_owned_tensors = sorted(OMIT_TARGET_OWNED_TENSORS - source_tensor_names)
    if missing_target_owned_tensors:
        raise RuntimeError(f"legacy artifact lacks expected target-owned tensors: {missing_target_owned_tensors}")

    for tensor in reader.tensors:
        if tensor.name in OMIT_TARGET_OWNED_TENSORS:
            continue
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
        if tensor.name in OMIT_TARGET_OWNED_TENSORS:
            continue
        writer.write_tensor_data(tensor.data, tensor_endianess=reader.endianess)
    writer.close()

    if not args.output.is_file() or args.output.stat().st_size <= 0:
        raise RuntimeError("canonicalized output was not created")
    print(f"created {args.output} ({args.output.stat().st_size} bytes)")
    print(f"added {CANONICAL_KEY}={target_layers}")
    print(f"added {TARGET_HIDDEN_SIZE_KEY}={target_hidden_size}")
    print(f"added {CANONICAL_MASK_KEY}={mask_token_id}")
    print(f"copied {len(target_tokenizer_fields)} tokenizer fields from {args.target_model}")
    print(f"renamed tensors: {TENSOR_RENAMES}")
    print(f"omitted target-owned tensors: {sorted(OMIT_TARGET_OWNED_TENSORS)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
