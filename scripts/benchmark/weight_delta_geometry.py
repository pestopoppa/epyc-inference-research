#!/usr/bin/env python3
"""Stream Q8_0 GGUF deltas without loading a model or starting inference.

The instrument compares stock Qwen3.6-27B weights with ThinkingCap and
Fable-Fusion.  It is deliberately opt-in: ``--plan`` is the default, while
``--execute`` is required before the process reads model tensor payloads.
"""

from __future__ import annotations

import argparse
import json
import math
import mmap
import struct
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Iterable, Sequence

import numpy as np


GGUF_MAGIC = 0x46554747
GGUF_VERSION = {2, 3}
GGUF_DEFAULT_ALIGNMENT = 32
GGML_TYPE_Q8_0 = 8
Q8_BLOCK_ELEMENTS = 32
Q8_BLOCK_BYTES = 34

DEFAULT_STOCK = Path("/mnt/raid0/llm/models/Qwen_Qwen3.6-27B-Q8_0.gguf")
DEFAULT_THINKINGCAP = Path(
    "/mnt/raid0/llm/models/ThinkingCap-Qwen3.6-27B-GGUF/"
    "ThinkingCap-Qwen3.6-27B-Q8_0.gguf"
)
DEFAULT_FABLE = Path(
    "/mnt/raid0/llm/models/Qwen3.6-27B-Fable-Fusion-711-GGUF/"
    "Qwen3.6-27B-Fable-Fus-711-UnHeretic-NM-DAU-NEO-MAX-NEO-Q8_0.gguf"
)


@dataclass(frozen=True)
class Tensor:
    name: str
    shape: tuple[int, ...]
    type_id: int
    offset: int

    @property
    def elements(self) -> int:
        return math.prod(self.shape) if self.shape else 1

    @property
    def bytes(self) -> int:
        if self.type_id != GGML_TYPE_Q8_0 or self.elements % Q8_BLOCK_ELEMENTS:
            raise ValueError(f"{self.name}: expected Q8_0 with a whole number of blocks")
        return self.elements // Q8_BLOCK_ELEMENTS * Q8_BLOCK_BYTES


@dataclass(frozen=True)
class Header:
    path: Path
    data_start: int
    alignment: int
    tensors: dict[str, Tensor]


def _read_exact(handle: BinaryIO, size: int) -> bytes:
    value = handle.read(size)
    if len(value) != size:
        raise ValueError("truncated GGUF header")
    return value


def _u32(handle: BinaryIO) -> int:
    return struct.unpack("<I", _read_exact(handle, 4))[0]


def _u64(handle: BinaryIO) -> int:
    return struct.unpack("<Q", _read_exact(handle, 8))[0]


def _string(handle: BinaryIO) -> str:
    return _read_exact(handle, _u64(handle)).decode("utf-8")


def _skip_value(handle: BinaryIO, value_type: int) -> None:
    scalar_sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
    if value_type == 8:
        handle.seek(_u64(handle), 1)
        return
    if value_type == 9:
        element_type, count = _u32(handle), _u64(handle)
        for _ in range(count):
            _skip_value(handle, element_type)
        return
    try:
        handle.seek(scalar_sizes[value_type], 1)
    except KeyError as exc:
        raise ValueError(f"unsupported GGUF metadata type {value_type}") from exc


def _align(offset: int, alignment: int) -> int:
    return (offset + alignment - 1) // alignment * alignment


def read_header(path: Path) -> Header:
    """Read only the GGUF header/tensor table; never materialize model payloads."""
    with path.open("rb") as handle:
        if _u32(handle) != GGUF_MAGIC:
            raise ValueError(f"{path}: not a GGUF file")
        if _u32(handle) not in GGUF_VERSION:
            raise ValueError(f"{path}: unsupported GGUF version")
        tensor_count, kv_count = _u64(handle), _u64(handle)
        alignment = GGUF_DEFAULT_ALIGNMENT
        for _ in range(kv_count):
            key, value_type = _string(handle), _u32(handle)
            if key == "general.alignment" and value_type == 4:
                alignment = _u32(handle)
            else:
                _skip_value(handle, value_type)
        tensors: dict[str, Tensor] = {}
        for _ in range(tensor_count):
            name, dimensions = _string(handle), _u32(handle)
            shape = tuple(_u64(handle) for _ in range(dimensions))
            tensor = Tensor(name, shape, _u32(handle), _u64(handle))
            if name in tensors:
                raise ValueError(f"{path}: duplicate tensor {name}")
            tensors[name] = tensor
        return Header(path, _align(handle.tell(), alignment), alignment, tensors)


def layer_for(name: str) -> str:
    parts = name.split(".")
    try:
        return f"blk.{int(parts[parts.index('blk') + 1])}"
    except (ValueError, IndexError):
        return "non_layer"


def q8_dequantize(raw: bytes) -> np.ndarray:
    """Dequantize GGML Q8_0 blocks (fp16 scale followed by 32 signed bytes)."""
    if len(raw) % Q8_BLOCK_BYTES:
        raise ValueError("Q8_0 payload is not block aligned")
    blocks = np.frombuffer(raw, dtype=np.uint8).reshape(-1, Q8_BLOCK_BYTES)
    scales = blocks[:, :2].copy().view("<f2").astype(np.float32)
    values = blocks[:, 2:].view(np.int8).astype(np.float32)
    return (values * scales[:, None]).reshape(-1)


def _tensor_triplets(stock: Header, thinkingcap: Header, fable: Header) -> Iterable[tuple[Tensor, Tensor, Tensor]]:
    for name in sorted(set(stock.tensors) & set(thinkingcap.tensors) & set(fable.tensors)):
        trio = (stock.tensors[name], thinkingcap.tensors[name], fable.tensors[name])
        if all(tensor.type_id == GGML_TYPE_Q8_0 for tensor in trio) and len({tensor.shape for tensor in trio}) == 1:
            yield trio


def _accumulate_tensor(
    stock_map: mmap.mmap,
    tc_map: mmap.mmap,
    ff_map: mmap.mmap,
    stock_header: Header,
    tc_header: Header,
    ff_header: Header,
    stock_tensor: Tensor,
    tc_tensor: Tensor,
    ff_tensor: Tensor,
    *,
    chunk_bytes: int,
) -> tuple[float, float, float]:
    if chunk_bytes < Q8_BLOCK_BYTES:
        raise ValueError("chunk_bytes must contain at least one Q8_0 block")
    chunk_bytes -= chunk_bytes % Q8_BLOCK_BYTES
    norm_tc = norm_ff = dot = 0.0
    for relative in range(0, stock_tensor.bytes, chunk_bytes):
        size = min(chunk_bytes, stock_tensor.bytes - relative)
        base_start = stock_header.data_start + stock_tensor.offset + relative
        tc_start = tc_header.data_start + tc_tensor.offset + relative
        ff_start = ff_header.data_start + ff_tensor.offset + relative
        base = q8_dequantize(stock_map[base_start:base_start + size])
        tc_values = q8_dequantize(tc_map[tc_start:tc_start + size])
        ff_values = q8_dequantize(ff_map[ff_start:ff_start + size])
        delta_tc, delta_ff = tc_values - base, ff_values - base
        norm_tc += float(np.dot(delta_tc, delta_tc))
        norm_ff += float(np.dot(delta_ff, delta_ff))
        dot += float(np.dot(delta_tc, delta_ff))
    return norm_tc, norm_ff, dot


def _geometry(norm_tc: float, norm_ff: float, dot: float) -> dict[str, float | None]:
    if norm_tc == 0.0 or norm_ff == 0.0:
        return {"r": None if norm_tc == 0.0 else math.sqrt(norm_ff / norm_tc), "cos": None, "p": None}
    return {"r": math.sqrt(norm_ff / norm_tc), "cos": dot / math.sqrt(norm_tc * norm_ff), "p": dot / norm_tc}


def execute(stock_path: Path, tc_path: Path, ff_path: Path, *, chunk_bytes: int) -> dict[str, object]:
    stock_header, tc_header, ff_header = (read_header(path) for path in (stock_path, tc_path, ff_path))
    skipped = {"not_shared_or_q8_or_shape_mismatch": 0}
    triplets = list(_tensor_triplets(stock_header, tc_header, ff_header))
    shared_names = set(stock_header.tensors) & set(tc_header.tensors) & set(ff_header.tensors)
    skipped["not_shared_or_q8_or_shape_mismatch"] = len(shared_names) - len(triplets)
    layers: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])
    tensor_rows: list[dict[str, object]] = []
    with stock_path.open("rb") as stock_file, tc_path.open("rb") as tc_file, ff_path.open("rb") as ff_file:
        with mmap.mmap(stock_file.fileno(), 0, access=mmap.ACCESS_READ) as stock_map, mmap.mmap(tc_file.fileno(), 0, access=mmap.ACCESS_READ) as tc_map, mmap.mmap(ff_file.fileno(), 0, access=mmap.ACCESS_READ) as ff_map:
            for base, tc, ff in triplets:
                norm_tc, norm_ff, dot = _accumulate_tensor(
                    stock_map, tc_map, ff_map, stock_header, tc_header, ff_header,
                    base, tc, ff, chunk_bytes=chunk_bytes,
                )
                layer = layer_for(base.name)
                layers[layer][0] += norm_tc
                layers[layer][1] += norm_ff
                layers[layer][2] += dot
                layers[layer][3] += 1
                tensor_rows.append({"name": base.name, "layer": layer, "norm_tc_sq": norm_tc, "norm_ff_sq": norm_ff, "dot": dot, **_geometry(norm_tc, norm_ff, dot)})
    layer_rows = [{"layer": layer, "norm_tc_sq": values[0], "norm_ff_sq": values[1], "dot": values[2], "tensor_count": int(values[3]), **_geometry(values[0], values[1], values[2])} for layer, values in sorted(layers.items())]
    zero_tc = [row["name"] for row in tensor_rows if row["norm_tc_sq"] == 0.0]
    return {
        "schema": "epyc.weight_delta_geometry.v1",
        "method": "streaming Q8_0 dequantization; no llama binary, server, GPU, or model inference",
        "inputs": {"stock": str(stock_path), "thinkingcap": str(tc_path), "fable": str(ff_path)},
        "q8_block": {"elements": Q8_BLOCK_ELEMENTS, "bytes": Q8_BLOCK_BYTES},
        "chunk_bytes": chunk_bytes,
        "tensor_count": len(tensor_rows),
        "zero_tc_tensor_count": len(zero_tc),
        "zero_tc_tensor_names": zero_tc,
        "skipped": skipped,
        "layers": layer_rows,
        "tensors": tensor_rows,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stock", type=Path, default=DEFAULT_STOCK)
    parser.add_argument("--thinkingcap", type=Path, default=DEFAULT_THINKINGCAP)
    parser.add_argument("--fable", type=Path, default=DEFAULT_FABLE)
    parser.add_argument("--output", type=Path, help="required with --execute")
    parser.add_argument("--chunk-mib", type=int, default=64)
    parser.add_argument("--execute", action="store_true", help="read GGUF tensor payloads (otherwise emit only plan)")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    plan = {"schema": "epyc.weight_delta_geometry.plan.v1", "will_execute": bool(args.execute), "inputs": {"stock": str(args.stock), "thinkingcap": str(args.thinkingcap), "fable": str(args.fable)}, "chunk_mib": args.chunk_mib, "execution": "requires explicit --execute; never invokes llama binaries, servers, or GPU"}
    if not args.execute:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0
    if args.output is None:
        raise SystemExit("--output is required with --execute")
    if args.chunk_mib <= 0:
        raise SystemExit("--chunk-mib must be positive")
    result = execute(args.stock, args.thinkingcap, args.fable, chunk_bytes=args.chunk_mib * 1024 * 1024)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
