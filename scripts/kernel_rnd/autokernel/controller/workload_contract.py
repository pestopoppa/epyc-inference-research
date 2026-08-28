#!/usr/bin/env python3
"""The screening workload must dispatch the kernels production dispatches.

WHY THIS EXISTS
---------------
The loop screened on `Qwen2.5-Coder-Q4_K_M.gguf` for a month. Reading the GGUF
header rather than the filename:

    n_embd = 896        896 / 256 = 3.5  -- NOT a whole number of K-quant superblocks
    Q5_0 x 132          every attention and FFN gate/up weight
    Q4_K x  12          twelve, out of 290 tensors

K-quants need a 256-element superblock, so llama.cpp silently fell back to **Q5_0**
for nearly the whole model. The kernel trace agrees: `mul_mat_vec_q<(ggml_type)6>`,
13,803 calls. Production serves 122B IQ2 and 27B-class models whose hidden dims ARE
divisible by 256 and therefore dispatch Q4_K / Q6_K / IQ2 kernels.

So the loop spent a month optimising a legacy quant path **production never
dispatches**, and its flagship hypothesis -- proposed 38 times -- was
`akh-v2-q5-type-specific-dequant`. A win there had no production surface to land on.
CH-6 measured the transfer gap independently: `MMQ_MFMA` OFF-vs-ON is +23.09% on
that model and +0.50% on Qwen3.8-27B.

Nobody chose that workload's dispatch path. They chose a filename.

WHAT THIS DOES
--------------
Reads the actual tensor census out of the GGUF and REFUSES a workload whose dominant
quantisation is not in the production family. A screening surface may legitimately be
smaller and cheaper than production -- that is the point of screening -- but it may
not dispatch different kernels, because then a measurement on it is not evidence
about production at all.

This is a structural gate, not a note in a README. The defect it closes was invisible
for a month precisely because it lived in a filename.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import struct
from pathlib import Path
from typing import Any, Mapping

WORKLOAD_SCHEMA = "epyc.autokernel.gpu_workload_contract.v1"

#: GGML type ids -> names. Only the ones a workload realistically carries.
GGML_TYPE_NAMES: Mapping[int, str] = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0",
    10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K", 15: "Q8_K",
    16: "IQ2_XXS", 17: "IQ2_XS", 19: "IQ3_XXS", 20: "IQ1_S", 21: "IQ4_NL",
    23: "IQ3_S", 24: "IQ2_S", 25: "IQ4_XS", 30: "BF16",
}

#: Quantisations the production stack actually dispatches (K-quants and I-quants).
#: A screening workload dominated by anything outside this set measures a kernel
#: production never runs.
PRODUCTION_QUANT_FAMILY = frozenset({
    "Q2_K", "Q3_K", "Q4_K", "Q5_K", "Q6_K", "Q8_K",
    "IQ1_S", "IQ2_XXS", "IQ2_XS", "IQ2_S", "IQ3_XXS", "IQ3_S", "IQ4_NL", "IQ4_XS",
})

#: K-quant superblock. A hidden dimension not divisible by this cannot carry
#: K-quants, and llama.cpp will fall back silently.
K_SUPERBLOCK = 256


class WorkloadContractError(ValueError):
    """The workload does not dispatch production's kernels."""


@dataclass(frozen=True)
class WorkloadCensus:
    """What a GGUF actually contains, as opposed to what it is called."""

    path: str
    architecture: str
    n_embd: int | None
    tensor_types: Mapping[str, int]

    @property
    def dominant_quant(self) -> str | None:
        """The most common NON-float tensor type: the kernel this model exercises."""
        quantised = {name: count for name, count in self.tensor_types.items()
                     if name not in {"F32", "F16", "BF16"}}
        if not quantised:
            return None
        return max(quantised.items(), key=lambda item: (item[1], item[0]))[0]

    @property
    def superblock_compatible(self) -> bool:
        return isinstance(self.n_embd, int) and self.n_embd % K_SUPERBLOCK == 0

    @property
    def in_production_family(self) -> bool:
        return self.dominant_quant in PRODUCTION_QUANT_FAMILY

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": WORKLOAD_SCHEMA,
            "path": self.path,
            "architecture": self.architecture,
            "n_embd": self.n_embd,
            "superblock_compatible": self.superblock_compatible,
            "dominant_quant": self.dominant_quant,
            "in_production_family": self.in_production_family,
            "tensor_types": dict(sorted(self.tensor_types.items())),
        }


def read_census(path: Path | str) -> WorkloadCensus:
    """Parse the GGUF header. The filename is not evidence; the tensor table is."""
    path = Path(path)
    with path.open("rb") as handle:
        if handle.read(4) != b"GGUF":
            raise WorkloadContractError(f"{path} is not a GGUF file")
        struct.unpack("<I", handle.read(4))                       # version
        tensor_count = struct.unpack("<Q", handle.read(8))[0]
        kv_count = struct.unpack("<Q", handle.read(8))[0]

        def read_string() -> str:
            length = struct.unpack("<Q", handle.read(8))[0]
            return handle.read(length).decode("utf-8", "replace")

        def read_value(type_id: int) -> Any:
            if type_id in (0, 7):
                return struct.unpack("<B", handle.read(1))[0]
            if type_id == 1:
                return struct.unpack("<b", handle.read(1))[0]
            if type_id == 2:
                return struct.unpack("<H", handle.read(2))[0]
            if type_id == 3:
                return struct.unpack("<h", handle.read(2))[0]
            if type_id == 4:
                return struct.unpack("<I", handle.read(4))[0]
            if type_id == 5:
                return struct.unpack("<i", handle.read(4))[0]
            if type_id == 6:
                return struct.unpack("<f", handle.read(4))[0]
            if type_id == 8:
                return read_string()
            if type_id == 9:
                element_type = struct.unpack("<I", handle.read(4))[0]
                length = struct.unpack("<Q", handle.read(8))[0]
                return [read_value(element_type) for _ in range(length)]
            if type_id == 10:
                return struct.unpack("<Q", handle.read(8))[0]
            if type_id == 11:
                return struct.unpack("<q", handle.read(8))[0]
            if type_id == 12:
                return struct.unpack("<d", handle.read(8))[0]
            raise WorkloadContractError(f"unknown GGUF metadata type {type_id}")

        metadata: dict[str, Any] = {}
        for _ in range(kv_count):
            key = read_string()
            metadata[key] = read_value(struct.unpack("<I", handle.read(4))[0])

        counts: Counter[str] = Counter()
        for _ in range(tensor_count):
            read_string()
            dims = struct.unpack("<I", handle.read(4))[0]
            for _ in range(dims):
                struct.unpack("<Q", handle.read(8))
            type_id = struct.unpack("<I", handle.read(4))[0]
            struct.unpack("<Q", handle.read(8))                   # offset
            counts[GGML_TYPE_NAMES.get(type_id, f"type_{type_id}")] += 1

    architecture = str(metadata.get("general.architecture", "unknown"))
    n_embd = metadata.get(f"{architecture}.embedding_length")
    return WorkloadCensus(
        path=str(path), architecture=architecture,
        n_embd=n_embd if isinstance(n_embd, int) else None,
        tensor_types=dict(counts))


def verify_workload(path: Path | str) -> WorkloadCensus:
    """Census the workload, or refuse it with the reason.

    Two independent refusals, because they fail differently:

      * a hidden dimension not divisible by 256 means K-quants CANNOT be used and
        llama.cpp will fall back silently -- the failure is invisible in the filename
        and in every log;
      * a dominant quant outside the production family means the screen exercises a
        kernel production never dispatches, whatever the hidden dimension is.
    """
    census = read_census(path)
    if not census.superblock_compatible:
        raise WorkloadContractError(
            f"{census.path}: n_embd={census.n_embd} is not divisible by "
            f"{K_SUPERBLOCK}, so K-quants cannot be used and llama.cpp falls back "
            f"silently. Observed dominant tensor type: {census.dominant_quant}. "
            f"This is exactly the Qwen2.5-Coder-0.5B defect: a file named Q4_K_M "
            f"that is 132x Q5_0 and 12x Q4_K.")
    if not census.in_production_family:
        raise WorkloadContractError(
            f"{census.path}: dominant quantisation {census.dominant_quant} is "
            f"outside the production family {sorted(PRODUCTION_QUANT_FAMILY)}. A "
            f"screening workload may be smaller than production, but it may not "
            f"dispatch different kernels -- a measurement on it is not evidence "
            f"about production.")
    return census


def write_minimal_gguf(path: Path | str, *, architecture: str = "qwen2",
                       n_embd: int = 1536,
                       tensor_types: Mapping[str, int] | None = None) -> Path:
    """Write a real, parseable GGUF header. For fixtures and parser round-trips.

    Deliberately a real header rather than a stub: a fixture that writes
    `model.json` and calls it a model is how a workload nobody had censused stayed
    in place for a month. A test that wants a model should produce something this
    module can actually read.
    """
    path = Path(path)
    counts = dict(tensor_types or {"Q4_K": 169, "F32": 141, "Q6_K": 29})
    ids = {name: type_id for type_id, name in GGML_TYPE_NAMES.items()}
    total = sum(counts.values())

    blob = bytearray(b"GGUF")
    blob += struct.pack("<I", 3)
    blob += struct.pack("<Q", total)
    blob += struct.pack("<Q", 2)

    def _string_kv(key: str, value: str) -> bytes:
        return (struct.pack("<Q", len(key)) + key.encode()
                + struct.pack("<I", 8)
                + struct.pack("<Q", len(value)) + value.encode())

    def _uint32_kv(key: str, value: int) -> bytes:
        return (struct.pack("<Q", len(key)) + key.encode()
                + struct.pack("<I", 4) + struct.pack("<I", value))

    blob += _string_kv("general.architecture", architecture)
    blob += _uint32_kv(f"{architecture}.embedding_length", n_embd)

    index = 0
    for name, count in counts.items():
        if name not in ids:
            raise WorkloadContractError(f"unknown tensor type name {name!r}")
        for _ in range(count):
            tensor_name = f"blk.{index}.weight".encode()
            index += 1
            blob += struct.pack("<Q", len(tensor_name)) + tensor_name
            blob += struct.pack("<I", 1) + struct.pack("<Q", 4096)
            blob += struct.pack("<I", ids[name]) + struct.pack("<Q", 0)

    path.write_bytes(bytes(blob))
    return path


__all__ = ["GGML_TYPE_NAMES", "K_SUPERBLOCK", "PRODUCTION_QUANT_FAMILY",
           "WORKLOAD_SCHEMA", "WorkloadCensus", "WorkloadContractError",
           "read_census", "verify_workload", "write_minimal_gguf"]
