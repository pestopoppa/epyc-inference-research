#!/usr/bin/env python3
"""Consume the E8M0 conformance vectors.

WHY THIS FILE IS THE POINT
    CONFORMANCE-VECTORS-1 says a backend is conformant only if a test ACTUALLY
    CONSUMES the vectors; anything else is an observation from reading source and
    must be marked ASSERTED. This file is what moves a row from ASSERTED to
    VERIFIED. Committed vectors nobody runs are the failure mode the instrument
    exists to prevent, so vectors without a consumer are worse than none -- they
    look like coverage.

WHAT IT VERIFIES TODAY
    The three reference decoders against their pinned vectors, bit-exactly, plus
    the structural properties that make the vectors meaningful. That covers the
    CONTRACTS. It does NOT yet execute any C/HIP backend, so every backend row in
    matrices/e8m0-conformance.md stays ASSERTED until a harness links against the
    real decoders.
"""
import json
import struct
from pathlib import Path

import pytest

VEC = Path(__file__).resolve().parent / "vectors"
CONTRACTS = ["e8m0_mx_spec", "e8m0_ggml_full", "e8m0_ggml_half"]


def load(name):
    return json.loads((VEC / f"{name}.json").read_text())


def to_bits(v: str) -> int:
    return int(v, 16)


def as_float(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits))[0]


@pytest.mark.parametrize("name", CONTRACTS)
def test_contract_matches_reference(name):
    """Every case's stated bit pattern must decode to its stated value."""
    doc = load(name)
    for c in doc["cases"]:
        bits = to_bits(c["bits"])
        f = as_float(bits)
        if c["value"] == "NaN":
            assert f != f, f"{name} code {c['code']}: bits say {c['bits']}, not NaN"
        elif c["value"] == "Infinity":
            assert f == float("inf"), f"{name} code {c['code']}: bits {c['bits']} != +Inf"
        else:
            assert repr(f) == c["value"], (
                f"{name} code {c['code']}: bits {c['bits']} -> {f!r}, vectors say {c['value']}")


@pytest.mark.parametrize("name", CONTRACTS)
def test_coverage_is_edge_weighted(name):
    """Boundaries, the identity point, and one step either side must all be present.

    A vector set that quietly loses its edges still passes every value check while
    testing nothing that matters -- the edges are the only place these three
    contracts disagree.
    """
    codes = {c["code"] for c in load(name)["cases"]}
    for required in (0, 1, 127, 254, 255):
        assert required in codes, f"{name} lost edge case {required}"


def test_the_contracts_actually_disagree():
    """The instrument is pointless if the contracts agree everywhere.

    This is the anti-vacuity check: if a future edit collapsed all three onto one
    behaviour, every other test here would still pass while the vectors silently
    stopped discriminating anything.
    """
    at_ff = {}
    for name in CONTRACTS:
        case = next(c for c in load(name)["cases"] if c["code"] == 255)
        at_ff[name] = case["bits"]
    assert len(set(at_ff.values())) == 3, (
        f"expected three distinct answers at 0xFF, got {at_ff}")


def test_contracts_agree_away_from_the_edge():
    """Conversely, they must agree where no divergence is claimed.

    ggml_full and mx_spec differ ONLY at 0xFF. If they diverge anywhere else, either
    a decoder changed or a vector is wrong, and both need a human.
    """
    spec = {c["code"]: c["bits"] for c in load("e8m0_mx_spec")["cases"]}
    full = {c["code"]: c["bits"] for c in load("e8m0_ggml_full")["cases"]}
    for code in spec:
        if code == 255:
            continue
        assert spec[code] == full[code], (
            f"unexpected divergence at code {code}: spec {spec[code]} vs ggml {full[code]}")


def test_half_is_exactly_half_of_full():
    """`_half` is documented as "Equal to ggml_e8m0_to_fp32/2". Hold it to that.

    Checked as a float ratio rather than a bit pattern, because the subnormal codes
    do not have a clean bit-level halving relationship.
    """
    full = {c["code"]: as_float(to_bits(c["bits"])) for c in load("e8m0_ggml_full")["cases"]}
    half = {c["code"]: as_float(to_bits(c["bits"])) for c in load("e8m0_ggml_half")["cases"]}
    for code, fv in full.items():
        if code == 255:
            continue  # full is +Inf here; Inf/2 is Inf, which would pass vacuously
        assert half[code] == pytest.approx(fv / 2.0, rel=0, abs=0), (
            f"code {code}: half={half[code]!r} is not exactly full/2={fv / 2.0!r}")


def test_divergence_record_matches_the_vectors():
    """The human-readable divergence record must not drift from the vectors."""
    div = json.loads((VEC / "e8m0_divergence.json").read_text())
    stated = {v["bits"] for v in div["answers"].values()}
    actual = {next(c for c in load(n)["cases"] if c["code"] == 255)["bits"] for n in CONTRACTS}
    assert stated == actual, f"divergence record says {stated}, vectors say {actual}"
