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

The production family is CENSUSED from the declared production model, never
hard-coded (2026-09-01, §5.1 of docs/design/autokernel-production-shaped-rung.md).
The hard-coded set this replaced was written in the K/I-quant era and never followed
the 2026-08-14 Q8_0 cutover, so `verify_workload` REFUSED production's own model --
the same drift class one level up. Reading the family off production's GGUF means
the next cutover cannot recur it.

`rung_matches_production` is the finer instrument on top: exact dominant-quant and
n_embd-class parity, required for a CONFIRM rung, recorded-but-WAIVED for a SCREEN
rung -- the waiver is a visible artifact, never a silent assumption.

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

#: Float carriers -- present in every GGUF, never the kernel a workload exercises.
FLOAT_TYPES = frozenset({"F32", "F16", "BF16"})

#: Superblock quants (K-quants and I-quants). These are DELIBERATE quantisation
#: choices: llama.cpp's silent-fallback path lands on legacy quants, never on these,
#: so a superblock-dominant workload is never the Qwen2.5-Coder accident. They stay
#: admissible as SCREEN workloads whatever production's dominant is -- the quant-axis
#: mismatch a screen carries is `rung_matches_production`'s recorded waiver, not this
#: gate's refusal. (This set was previously named PRODUCTION_QUANT_FAMILY and WAS the
#: whole gate; the production family is now censused, see `production_quant_family`.)
SUPERBLOCK_QUANT_FAMILY = frozenset({
    "Q2_K", "Q3_K", "Q4_K", "Q5_K", "Q6_K", "Q8_K",
    "IQ1_S", "IQ2_XXS", "IQ2_XS", "IQ2_S", "IQ3_XXS", "IQ3_S", "IQ4_NL", "IQ4_XS",
})

#: The DECLARED production model -- the reference GGUF the family is censused from.
#: Default documented here because `run.py` threads no path yet: Qwen3.8-27B-Q8_0 is
#: the 2026-08-14 cutover's serving model (arch qwen35, n_embd 5120, dominant Q8_0,
#: censused -- not read off the filename). A promotion that changes the serving model
#: updates THIS constant or passes `production_model` explicitly.
PRODUCTION_MODEL = Path("/mnt/raid0/llm/models/Qwen3.8-27B-Q8_0.gguf")

#: Rung roles for `rung_matches_production`.
SCREEN_RUNG = "screen"
CONFIRM_RUNG = "confirm"

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
                     if name not in FLOAT_TYPES}
        if not quantised:
            return None
        return max(quantised.items(), key=lambda item: (item[1], item[0]))[0]

    @property
    def superblock_compatible(self) -> bool:
        return isinstance(self.n_embd, int) and self.n_embd % K_SUPERBLOCK == 0

    @property
    def quantised_types(self) -> frozenset[str]:
        """Every non-float tensor type this model carries: the kernels it dispatches."""
        return frozenset(name for name in self.tensor_types
                         if name not in FLOAT_TYPES)

    def to_dict(self) -> dict[str, Any]:
        # `in_production_family` left this record 2026-09-01: a census cannot know
        # the family by itself any more -- membership is a two-census fact
        # (`verify_workload` / `rung_matches_production`), and a stale boolean here
        # would be the drift this change exists to close.
        return {
            "schema": WORKLOAD_SCHEMA,
            "path": self.path,
            "architecture": self.architecture,
            "n_embd": self.n_embd,
            "superblock_compatible": self.superblock_compatible,
            "dominant_quant": self.dominant_quant,
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


def production_census(production_model: Path | str = PRODUCTION_MODEL
                      ) -> WorkloadCensus:
    """Census the declared production model, or refuse LOUDLY.

    Never fail-open: a production GGUF that cannot be read must refuse every
    workload with a message naming why, because a silent pass here reopens the
    filename-trust hole one level up -- the loop would verify workloads against a
    family nobody censused.
    """
    try:
        return read_census(production_model)
    except (OSError, WorkloadContractError) as exc:
        raise WorkloadContractError(
            f"cannot census the declared production model {production_model}: {exc}. "
            f"Refusing to verify ANY workload -- with no production census there is "
            f"no production family to check against, and passing on a guess is the "
            f"drift this census replaced the hard-coded family to prevent. Fix the "
            f"production model path (PRODUCTION_MODEL) or pass production_model "
            f"explicitly.") from exc


def production_quant_family(production: WorkloadCensus) -> frozenset[str]:
    """The admissible dominant quants, censused from production's own tensor table.

    Superblock quants (never fallback artifacts; screens keep quant-axis latitude,
    with the mismatch recorded by `rung_matches_production`) plus every quantised
    type production actually carries -- which is what admits a deliberate
    legacy-family production model like Q8_0 without re-admitting the silent-fallback
    targets (Q5_0, Q4_1, ...) production does not dispatch.
    """
    carried = production.quantised_types
    if not carried:
        raise WorkloadContractError(
            f"{production.path}: the declared production model carries no quantised "
            f"tensors at all ({dict(production.tensor_types)}); it cannot define a "
            f"production quant family, and no workload can be verified against it")
    return SUPERBLOCK_QUANT_FAMILY | carried


def verify_workload(path: Path | str, *,
                    production_model: Path | str = PRODUCTION_MODEL,
                    production: WorkloadCensus | None = None) -> WorkloadCensus:
    """Census the workload, or refuse it with the reason.

    Two independent refusals, because they fail differently:

      * a hidden dimension not divisible by 256 means K-quants CANNOT be used and
        llama.cpp will fall back silently -- the failure is invisible in the filename
        and in every log;
      * a dominant quant outside the CENSUSED production family means the workload
        exercises a kernel production never dispatches, whatever the hidden
        dimension is.

    `production` short-circuits the production census when the caller already holds
    one (tests, and callers verifying several workloads against one reference).
    """
    census = read_census(path)
    if not census.superblock_compatible:
        raise WorkloadContractError(
            f"{census.path}: n_embd={census.n_embd} is not divisible by "
            f"{K_SUPERBLOCK}, so K-quants cannot be used and llama.cpp falls back "
            f"silently. Observed dominant tensor type: {census.dominant_quant}. "
            f"This is exactly the Qwen2.5-Coder-0.5B defect: a file named Q4_K_M "
            f"that is 132x Q5_0 and 12x Q4_K.")
    reference = production if production is not None \
        else production_census(production_model)
    family = production_quant_family(reference)
    if census.dominant_quant not in family:
        raise WorkloadContractError(
            f"{census.path}: dominant quantisation {census.dominant_quant} is "
            f"outside the production family {sorted(family)}, censused from "
            f"{reference.path} (dominant {reference.dominant_quant}). A workload "
            f"may be smaller than production, but it may not dispatch different "
            f"kernels -- a measurement on it is not evidence about production.")
    return census


@dataclass(frozen=True)
class RungParity:
    """Whether a rung's workload matches production's dispatch shape, structurally.

    `exact` is dominant-quant equality AND n_embd-class equality. A CONFIRM rung
    must be exact -- its whole job is gating keeps on the production shape. A SCREEN
    rung that is not exact gets `waived=True`: the run proceeds, but the mismatch is
    a visible artifact in this record, never a silent assumption -- R23-5 is what a
    silent one costs (+17.26% at b1 that is -1.46% on the production shape).
    """

    rung: str
    workload: str
    production: str
    workload_dominant: str | None
    production_dominant: str | None
    workload_n_embd: int | None
    production_n_embd: int | None
    dominant_quant_match: bool
    n_embd_class_match: bool
    exact: bool
    waived: bool
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "epyc.autokernel.rung_parity.v1",
            "rung": self.rung, "workload": self.workload,
            "production": self.production,
            "workload_dominant": self.workload_dominant,
            "production_dominant": self.production_dominant,
            "workload_n_embd": self.workload_n_embd,
            "production_n_embd": self.production_n_embd,
            "dominant_quant_match": self.dominant_quant_match,
            "n_embd_class_match": self.n_embd_class_match,
            "exact": self.exact, "waived": self.waived, "detail": self.detail,
        }


def rung_matches_production(census: WorkloadCensus, production: WorkloadCensus, *,
                            rung: str) -> RungParity:
    """Structured rung-vs-production parity. Never raises on a mismatch: the CALLER
    decides what a non-exact result means for its rung, and this record is what it
    decides on (and what it must persist).

    n_embd CLASS is the n_embd value itself: the width selects tile/launch geometry
    and whether `fixed-<width>` specializations dispatch at all, and no coarser
    bucketing has a measured basis (4096 vs 5120 was scored "near" and still cannot
    dispatch a fixed-5120 specialization).
    """
    if rung not in (SCREEN_RUNG, CONFIRM_RUNG):
        raise WorkloadContractError(
            f"unknown rung role {rung!r}: expected {SCREEN_RUNG!r} or {CONFIRM_RUNG!r}")
    quant_match = (census.dominant_quant is not None
                   and census.dominant_quant == production.dominant_quant)
    width_match = (isinstance(census.n_embd, int)
                   and census.n_embd == production.n_embd)
    exact = quant_match and width_match
    waived = (not exact) and rung == SCREEN_RUNG
    detail = (f"{rung} rung {census.path}: dominant "
              f"{census.dominant_quant}{'==' if quant_match else '!='}"
              f"{production.dominant_quant}, n_embd "
              f"{census.n_embd}{'==' if width_match else '!='}"
              f"{production.n_embd} vs production "
              + ("-- EXACT" if exact else
                 ("-- MISMATCH WAIVED for the screen rung: measurements here are "
                  "null-killers, not production evidence" if waived else
                  "-- MISMATCH, not waivable for a confirm rung")))
    return RungParity(
        rung=rung, workload=census.path, production=production.path,
        workload_dominant=census.dominant_quant,
        production_dominant=production.dominant_quant,
        workload_n_embd=census.n_embd, production_n_embd=production.n_embd,
        dominant_quant_match=quant_match, n_embd_class_match=width_match,
        exact=exact, waived=waived, detail=detail)


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


__all__ = ["CONFIRM_RUNG", "FLOAT_TYPES", "GGML_TYPE_NAMES", "K_SUPERBLOCK",
           "PRODUCTION_MODEL", "RungParity", "SCREEN_RUNG",
           "SUPERBLOCK_QUANT_FAMILY", "WORKLOAD_SCHEMA", "WorkloadCensus",
           "WorkloadContractError", "production_census", "production_quant_family",
           "read_census", "rung_matches_production", "verify_workload",
           "write_minimal_gguf"]
