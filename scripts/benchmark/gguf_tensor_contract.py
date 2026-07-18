#!/usr/bin/env python3
"""Dump selected GGUF tensor and metadata descriptors without loading a model.

This is intended for kernel-prep audits such as GLM NextN/MTP tail-block mapping.
It uses gguf-py's header/tensor descriptor reader and does not start inference.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import struct
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence


DEFAULT_GGUF_PY = Path("/mnt/raid0/llm/llama.cpp-experimental/gguf-py")
GLM_NEXTN_METADATA_REGEX = r"general\.architecture|glm-dsa\.|glm4-moe\."
GLM_NEXTN_CORE_SUFFIX_SHAPES = {
    "nextn.eh_proj.weight": lambda n_embd: [2 * n_embd, n_embd],
    "nextn.enorm.weight": lambda n_embd: [n_embd],
    "nextn.hnorm.weight": lambda n_embd: [n_embd],
}
GLM_NEXTN_OPTIONAL_SUFFIX_SHAPES = {
    "nextn.shared_head_norm.weight": lambda n_embd: [n_embd],
}
GGUF_MAGIC = 0x46554747
GGUF_DEFAULT_ALIGNMENT = 32
GGUF_VALUE_SCALAR_SIZES = {
    0: 1,   # UINT8
    1: 1,   # INT8
    2: 2,   # UINT16
    3: 2,   # INT16
    4: 4,   # UINT32
    5: 4,   # INT32
    6: 4,   # FLOAT32
    7: 1,   # BOOL
    10: 8,  # UINT64
    11: 8,  # INT64
    12: 8,  # FLOAT64
}
GGML_QUANT_SIZE_BY_ID = {
    0: ("F32", 1, 4),
    1: ("F16", 1, 2),
    2: ("Q4_0", 32, 18),
    8: ("Q8_0", 32, 34),
    30: ("BF16", 1, 2),
    35: ("TQ2_0", 256, 66),
    41: ("Q1_0", 128, 18),
    42: ("Q2_0", 64, 18),
}


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


@dataclass(frozen=True)
class RawGgufTensorInfo:
    name: str
    shape: list[int]
    n_elements: int
    type_id: int
    type_name: str
    data_offset: int
    expected_nbytes: int | None
    expected_span: int | None
    physical_span: int | None
    span_delta: int | None


@dataclass(frozen=True)
class RawGgufHeader:
    file: str
    file_bytes: int
    version: int
    tensor_count: int
    kv_count: int
    alignment: int
    tensor_info_end: int
    data_start: int
    metadata: dict[str, Any]
    tensors: list[RawGgufTensorInfo]


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


def align_offset(offset: int, alignment: int) -> int:
    if alignment <= 0:
        raise ValueError(f"alignment must be positive, got {alignment}")
    return ((offset + alignment - 1) // alignment) * alignment


def read_u32(data: bytes, offset: int) -> tuple[int, int]:
    return struct.unpack_from("<I", data, offset)[0], offset + 4


def read_u64(data: bytes, offset: int) -> tuple[int, int]:
    return struct.unpack_from("<Q", data, offset)[0], offset + 8


def read_gguf_string(data: bytes, offset: int) -> tuple[str, int]:
    length, offset = read_u64(data, offset)
    end = offset + length
    if end > len(data):
        raise ValueError("truncated GGUF string")
    return data[offset:end].decode("utf-8"), end


def skip_gguf_value(
    data: bytes,
    offset: int,
    value_type: int,
    *,
    retain: bool = False,
) -> tuple[Any | None, int]:
    if value_type == 8:  # STRING
        value, offset = read_gguf_string(data, offset)
        return (value if retain else None), offset
    if value_type == 9:  # ARRAY
        element_type, offset = read_u32(data, offset)
        count, offset = read_u64(data, offset)
        values: list[Any] = []
        for _ in range(count):
            value, offset = skip_gguf_value(data, offset, element_type, retain=retain)
            if retain:
                values.append(value)
        return values if retain else None, offset
    size = GGUF_VALUE_SCALAR_SIZES.get(value_type)
    if size is None:
        raise ValueError(f"unsupported GGUF metadata value type {value_type}")
    raw = data[offset:offset + size]
    if len(raw) != size:
        raise ValueError("truncated GGUF metadata value")
    offset += size
    if value_type in {0, 1, 2, 3, 4, 5, 10, 11}:
        signed = value_type in {1, 3, 5, 11}
        value = int.from_bytes(raw, "little", signed=signed)
        return (value if retain else None), offset
    if value_type == 6:
        value = struct.unpack("<f", raw)[0]
        return (value if retain else None), offset
    if value_type == 12:
        value = struct.unpack("<d", raw)[0]
        return (value if retain else None), offset
    if value_type == 7:
        value = bool(raw[0])
        return (value if retain else None), offset
    return None, offset


def expected_ggml_nbytes(type_id: int, n_elements: int) -> int | None:
    quant = GGML_QUANT_SIZE_BY_ID.get(type_id)
    if quant is None:
        return None
    _name, block_size, type_size = quant
    return math.ceil(n_elements / block_size) * type_size


def read_raw_gguf_header(path: Path) -> RawGgufHeader:
    data = path.read_bytes()
    offset = 0
    magic, offset = read_u32(data, offset)
    if magic != GGUF_MAGIC:
        raise ValueError(f"GGUF magic invalid: 0x{magic:08x}")
    version, offset = read_u32(data, offset)
    if version not in {2, 3}:
        raise ValueError(f"unsupported GGUF version {version}")
    tensor_count, offset = read_u64(data, offset)
    kv_count, offset = read_u64(data, offset)

    metadata: dict[str, Any] = {}
    alignment = GGUF_DEFAULT_ALIGNMENT
    for _ in range(kv_count):
        key, offset = read_gguf_string(data, offset)
        value_type, offset = read_u32(data, offset)
        keep_value = key in {"general.architecture", "general.alignment"}
        value, offset = skip_gguf_value(data, offset, value_type, retain=keep_value)
        if key in {"general.architecture", "general.alignment"}:
            metadata[key] = value
        if key == "general.alignment" and isinstance(value, int):
            alignment = value

    tensor_records: list[dict[str, Any]] = []
    for _ in range(tensor_count):
        name, offset = read_gguf_string(data, offset)
        n_dims, offset = read_u32(data, offset)
        shape: list[int] = []
        for _dim in range(n_dims):
            dim, offset = read_u64(data, offset)
            shape.append(int(dim))
        type_id, offset = read_u32(data, offset)
        tensor_offset, offset = read_u64(data, offset)
        n_elements = math.prod(shape) if shape else 1
        expected = expected_ggml_nbytes(type_id, n_elements)
        tensor_records.append({
            "name": name,
            "shape": shape,
            "n_elements": n_elements,
            "type_id": type_id,
            "type_name": GGML_QUANT_SIZE_BY_ID.get(type_id, (f"TYPE_{type_id}", 1, 1))[0],
            "data_offset": int(tensor_offset),
            "expected_nbytes": expected,
        })

    tensor_info_end = offset
    data_start = align_offset(tensor_info_end, alignment)
    sorted_records = sorted(tensor_records, key=lambda item: int(item["data_offset"]))
    tensors: list[RawGgufTensorInfo] = []
    for idx, item in enumerate(sorted_records):
        expected_nbytes = item["expected_nbytes"]
        expected_span = (
            align_offset(expected_nbytes, alignment) if expected_nbytes is not None else None
        )
        if idx + 1 < len(sorted_records):
            physical_span = int(sorted_records[idx + 1]["data_offset"]) - int(item["data_offset"])
        else:
            physical_span = len(data) - data_start - int(item["data_offset"])
        span_delta = None
        if expected_span is not None and physical_span is not None:
            span_delta = physical_span - expected_span
        tensors.append(RawGgufTensorInfo(
            name=str(item["name"]),
            shape=[int(dim) for dim in item["shape"]],
            n_elements=int(item["n_elements"]),
            type_id=int(item["type_id"]),
            type_name=str(item["type_name"]),
            data_offset=int(item["data_offset"]),
            expected_nbytes=expected_nbytes,
            expected_span=expected_span,
            physical_span=physical_span,
            span_delta=span_delta,
        ))

    return RawGgufHeader(
        file=str(path),
        file_bytes=len(data),
        version=version,
        tensor_count=tensor_count,
        kv_count=kv_count,
        alignment=alignment,
        tensor_info_end=tensor_info_end,
        data_start=data_start,
        metadata=metadata,
        tensors=tensors,
    )


def validate_q2_layout_contract(files: Sequence[Path]) -> dict[str, Any]:
    """Validate Q2_0 physical tensor spans without gguf-py or llama.cpp loading."""
    file_reports: list[dict[str, Any]] = []
    errors: list[str] = []
    warnings: list[str] = []

    for file in files:
        header = read_raw_gguf_header(file)
        q2_tensors = [tensor for tensor in header.tensors if tensor.type_name == "Q2_0"]
        mismatches = [
            tensor for tensor in q2_tensors
            if tensor.span_delta is not None and tensor.span_delta != 0
        ]
        short = [tensor for tensor in mismatches if tensor.span_delta is not None and tensor.span_delta < 0]
        long = [tensor for tensor in mismatches if tensor.span_delta is not None and tensor.span_delta > 0]
        if short:
            first = short[0]
            errors.append(
                f"{file}: Q2_0 tensor {first.name} is {-int(first.span_delta)} bytes short "
                f"(physical {first.physical_span}, expected {first.expected_span})"
            )
        if long:
            warnings.append(
                f"{file}: {len(long)} Q2_0 tensor(s) have extra physical span; inspect before loader work"
            )

        file_reports.append({
            "file": header.file,
            "file_bytes": header.file_bytes,
            "version": header.version,
            "tensor_count": header.tensor_count,
            "kv_count": header.kv_count,
            "alignment": header.alignment,
            "tensor_info_end": header.tensor_info_end,
            "data_start": header.data_start,
            "metadata": header.metadata,
            "q2_0_tensor_count": len(q2_tensors),
            "q2_0_mismatch_count": len(mismatches),
            "q2_0_short_count": len(short),
            "q2_0_long_count": len(long),
            "first_q2_0_tensors": [asdict(tensor) for tensor in q2_tensors[:5]],
            "mismatches": [asdict(tensor) for tensor in mismatches[:20]],
        })

    return {
        "schema": "epyc.gguf_q2_layout_contract.v1",
        "passed": not errors,
        "errors": errors,
        "warnings": warnings,
        "assumptions": {
            "gguf_parser": "raw header/tensor-info parser; does not use gguf-py tensor array reshape or llama.cpp model loading",
            "q2_0_current_v7_layout": {"block_size": 64, "type_size": 18},
            "span_comparison": "next tensor offset minus current tensor offset compared to alignment-padded expected bytes",
        },
        "files": file_reports,
    }


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


def scalar_value(value: Any) -> Any:
    if isinstance(value, list) and len(value) == 1:
        return value[0]
    return value


def metadata_lookup(contract: dict[str, Any], key: str) -> Any | None:
    for file_metadata in contract.get("metadata", {}).values():
        if key in file_metadata:
            return scalar_value(file_metadata[key])
    return None


def int_metadata(contract: dict[str, Any], key: str, errors: list[str]) -> int | None:
    value = metadata_lookup(contract, key)
    if value is None:
        errors.append(f"missing metadata key: {key}")
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        errors.append(f"metadata key {key} is not an integer: {value!r}")
        return None


def tensor_by_name(contract: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(tensor["name"]): tensor for tensor in contract.get("tensors", [])}


def classify_glm_tail_tensor(name: str, tail_layers: set[int]) -> str | None:
    layer = layer_index(name)
    if layer not in tail_layers:
        return None
    if ".nextn." in name:
        return "nextn"
    if ".indexer." in name:
        return "indexer"
    if ".attn_" in name or ".attn." in name:
        return "attention"
    if ".ffn_" in name or ".exp_probs_" in name:
        return "ffn"
    return "other"


def validate_glm_nextn_contract(contract: dict[str, Any]) -> dict[str, Any]:
    """Validate the GLM physical NextN tail needed before a decoder-MTP port."""
    errors: list[str] = []
    warnings: list[str] = []

    arch = metadata_lookup(contract, "general.architecture")
    if arch is None:
        errors.append("missing metadata key: general.architecture")
        arch_prefix = "glm-dsa"
    else:
        arch = str(arch)
        arch_prefix = arch
        if arch not in {"glm-dsa", "glm4-moe"}:
            errors.append(f"unsupported GLM NextN architecture: {arch}")

    block_count = int_metadata(contract, f"{arch_prefix}.block_count", errors)
    nextn_layers = int_metadata(contract, f"{arch_prefix}.nextn_predict_layers", errors)
    n_embd = int_metadata(contract, f"{arch_prefix}.embedding_length", errors)

    facts: dict[str, Any] = {
        "architecture": arch,
        "block_count": block_count,
        "nextn_predict_layers": nextn_layers,
        "embedding_length": n_embd,
    }

    if block_count is None or nextn_layers is None or n_embd is None:
        return {"passed": False, "errors": errors, "warnings": warnings, "facts": facts}
    if nextn_layers <= 0:
        errors.append(f"{arch_prefix}.nextn_predict_layers must be > 0, got {nextn_layers}")
        return {"passed": False, "errors": errors, "warnings": warnings, "facts": facts}
    if nextn_layers >= block_count:
        errors.append(
            f"{arch_prefix}.nextn_predict_layers must be smaller than block_count "
            f"({nextn_layers} >= {block_count})"
        )
        return {"passed": False, "errors": errors, "warnings": warnings, "facts": facts}
    if nextn_layers != 1:
        warnings.append(
            "current qwen35-style decoder-MTP reference supports one NextN block; "
            f"artifact advertises {nextn_layers}"
        )

    tail_layers = list(range(block_count - nextn_layers, block_count))
    tail_layer_set = set(tail_layers)
    facts["tail_layers"] = tail_layers

    tensors = tensor_by_name(contract)
    group_counts = {"attention": 0, "ffn": 0, "indexer": 0, "nextn": 0, "other": 0}
    for name in tensors:
        group = classify_glm_tail_tensor(name, tail_layer_set)
        if group:
            group_counts[group] += 1
    facts["tail_group_counts"] = group_counts

    for tail_layer in tail_layers:
        prefix = f"blk.{tail_layer}."
        if not any(name.startswith(prefix) for name in tensors):
            errors.append(
                f"missing descriptors for physical NextN tail layer {tail_layer}; "
                "rerun with a tensor regex/layer range that includes the tail"
            )
            continue
        for suffix, expected_shape_fn in GLM_NEXTN_CORE_SUFFIX_SHAPES.items():
            name = prefix + suffix
            tensor = tensors.get(name)
            if tensor is None:
                errors.append(f"missing required NextN tensor: {name}")
                continue
            expected_shape = expected_shape_fn(n_embd)
            if list(tensor["shape"]) != expected_shape:
                errors.append(
                    f"wrong shape for {name}: expected {expected_shape}, got {tensor['shape']}"
                )
        for suffix, expected_shape_fn in GLM_NEXTN_OPTIONAL_SUFFIX_SHAPES.items():
            name = prefix + suffix
            tensor = tensors.get(name)
            if tensor is None:
                warnings.append(f"optional NextN tensor absent: {name}")
                continue
            expected_shape = expected_shape_fn(n_embd)
            if list(tensor["shape"]) != expected_shape:
                errors.append(
                    f"wrong shape for optional {name}: expected {expected_shape}, got {tensor['shape']}"
                )

    if arch == "glm-dsa":
        for group in ("attention", "ffn", "indexer"):
            if group_counts[group] == 0:
                errors.append(
                    f"glm-dsa tail has no {group} tensors in this descriptor set; "
                    "the MTP graph cannot be treated as a dense Qwen clone"
                )

    return {
        "passed": not errors,
        "errors": errors,
        "warnings": warnings,
        "facts": facts,
        "required_nextn_suffixes": sorted(GLM_NEXTN_CORE_SUFFIX_SHAPES),
        "optional_nextn_suffixes": sorted(GLM_NEXTN_OPTIONAL_SUFFIX_SHAPES),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path, help="GGUF file(s) or directories")
    parser.add_argument("--gguf-py", type=Path, default=DEFAULT_GGUF_PY)
    parser.add_argument(
        "--contract",
        choices=["glm-nextn", "q2-layout"],
        default=None,
        help="Run a fail-closed contract validator and include its result in the JSON.",
    )
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


def unique_patterns(patterns: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(patterns))


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    files = discover_gguf_files(args.paths)
    if not files:
        raise SystemExit("no GGUF files found")
    if args.contract == "q2-layout":
        contract = validate_q2_layout_contract(files)
        payload = json.dumps(contract, indent=2, sort_keys=True) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(payload)
        else:
            print(payload, end="")
        return 0 if contract["passed"] else 1

    metadata_patterns = list(args.metadata_regex)
    if args.contract == "glm-nextn":
        metadata_patterns.append(GLM_NEXTN_METADATA_REGEX)
    contract = read_contract(
        files,
        gguf_py=args.gguf_py,
        tensor_patterns=args.tensor_regex,
        metadata_patterns=unique_patterns(metadata_patterns),
        layer_start=args.layer_start,
        layer_end=args.layer_end,
    )
    if args.contract == "glm-nextn":
        contract["contract"] = {"glm_nextn": validate_glm_nextn_contract(contract)}
    payload = json.dumps(contract, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)
    else:
        print(payload, end="")
    if args.contract == "glm-nextn" and not contract["contract"]["glm_nextn"]["passed"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
