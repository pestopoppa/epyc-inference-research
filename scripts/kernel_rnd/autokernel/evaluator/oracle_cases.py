"""Loop-owned extra oracle cases for one target op (AK-GFX90A-ASM-SURFACE gap 3).

A candidate may never edit ``tests/test-backend-ops.cpp`` (``SourcePatchManifest.bind``
refuses reward-instrument translation units). A fast path with shape or mask
restrictions (e.g. the gfx90a FA hd128 asm path: causal/full masks only) is then never
reached by the stock corpus, so it would be "correct" without ever being checked.

The fix lives on the loop side: a case set is a ``test-backend-ops --test-file`` file
(the ``test_generic_op`` line format of ``make_test_cases_from_file``) committed in this
research repo next to a small JSON descriptor that pins its sha256, its target op and a
path witness. The candidate cannot author or alter it. A run is admitted as coverage
only when (a) every case line targets the declared op, (b) the bytes match the pin, and
(c) the witness regex appears in the run output (the fast path really ran). The witness
is necessary, not sufficient: correctness is still the oracle's own numeric verdict, and
the caller should pair it with a kill-switch run where the witness must be absent.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
import re
from typing import Mapping

__all__ = ["SCHEMA", "OracleCaseError", "OracleCaseSet", "REGISTRY_DIR",
           "ggml_op_ids", "load_case_set", "oracle_argv", "check_witness"]

SCHEMA = "epyc.autokernel.oracle_case_set.v1"
REGISTRY_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "oracle_cases")
MAX_CASES = 256
_OP_NAME = re.compile(r"^[A-Z][A-Z0-9_]*$")
_NAME = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class OracleCaseError(RuntimeError):
    """An oracle case set cannot be admitted as coverage."""


@dataclass(frozen=True)
class OracleCaseSet:
    op: str
    name: str
    cases_sha256: str
    n_cases: int
    witness: str
    cases_path: str

    def __post_init__(self) -> None:
        if not _OP_NAME.match(self.op) or not _NAME.match(self.name):
            raise OracleCaseError("case set op/name is malformed")
        if not _SHA256.match(self.cases_sha256):
            raise OracleCaseError("cases_sha256 must be a sha256 hex digest")
        if type(self.n_cases) is not int or not 1 <= self.n_cases <= MAX_CASES:
            raise OracleCaseError(f"n_cases must be an int in [1, {MAX_CASES}]")
        if not isinstance(self.witness, str) or not self.witness.strip():
            raise OracleCaseError("a case set must declare a non-empty path witness")
        try:
            re.compile(self.witness)
        except re.error as exc:
            raise OracleCaseError(f"witness is not a valid regex: {exc}") from exc


def ggml_op_ids(ggml_h_text: str) -> dict[str, int]:
    """GGML_OP_* name -> enum value, parsed from the tree under test's ggml.h."""
    head = re.search(r"\bGGML_OP_NONE\s*=\s*0\s*,", ggml_h_text)
    start = head.start() if head else -1
    end = ggml_h_text.find("GGML_OP_COUNT,", start)
    if start < 0 or end < 0:
        raise OracleCaseError("ggml.h has no GGML_OP enum")
    out, value = {}, -1
    for name, explicit in re.findall(r"\b(GGML_OP_[A-Z0-9_]+)\s*(?:=\s*(\d+))?\s*,",
                                     ggml_h_text[start:end + len("GGML_OP_COUNT,")]):
        value = int(explicit) if explicit else value + 1
        out[name[len("GGML_OP_"):]] = value
    if "NONE" not in out or out["NONE"] != 0:
        raise OracleCaseError("ggml.h GGML_OP enum did not parse")
    return out


def load_case_set(descriptor_path: str, *, ggml_h_text: str) -> OracleCaseSet:
    """Read a descriptor + its case file, fail closed on any mismatch."""
    real = os.path.realpath(descriptor_path)
    try:
        with open(real, encoding="utf-8") as handle:
            desc = json.load(handle)
    except (OSError, ValueError) as exc:
        raise OracleCaseError(f"case set descriptor unreadable: {exc}") from exc
    keys = {"schema", "op", "name", "cases_file", "cases_sha256", "n_cases", "witness"}
    if not isinstance(desc, Mapping) or set(desc) != keys or desc["schema"] != SCHEMA:
        raise OracleCaseError(f"descriptor fields must be exactly {sorted(keys)} with schema {SCHEMA}")
    base = os.path.dirname(real)
    cases_file = desc["cases_file"]
    if not isinstance(cases_file, str) or os.path.basename(cases_file) != cases_file:
        raise OracleCaseError("cases_file must be a bare filename next to the descriptor")
    cases_path = os.path.join(base, cases_file)
    try:
        data = open(cases_path, "rb").read()
    except OSError as exc:
        raise OracleCaseError(f"case file unreadable: {exc}") from exc
    case_set = OracleCaseSet(op=desc["op"], name=desc["name"], cases_sha256=desc["cases_sha256"],
                             n_cases=desc["n_cases"], witness=desc["witness"], cases_path=cases_path)
    if hashlib.sha256(data).hexdigest() != case_set.cases_sha256:
        raise OracleCaseError("case file bytes do not match cases_sha256")
    ids = ggml_op_ids(ggml_h_text)
    if case_set.op not in ids:
        raise OracleCaseError(f"op {case_set.op!r} is not in the tree's GGML_OP enum")
    lines = [line for line in data.decode("ascii").splitlines() if line.strip()]
    if len(lines) != case_set.n_cases:
        raise OracleCaseError(f"case file has {len(lines)} cases, descriptor says {case_set.n_cases}")
    for index, line in enumerate(lines):
        first = line.split()[0]
        if not first.isdigit() or int(first) != ids[case_set.op]:
            raise OracleCaseError(f"case {index} targets op id {first}, not {case_set.op}")
    return case_set


def oracle_argv(case_set: OracleCaseSet, *, backend: str, mode: str = "test") -> list[str]:
    """test-backend-ops arguments (after the binary) that run exactly this case set."""
    if mode not in ("test", "perf") or not re.match(r"^[A-Za-z0-9_]+$", backend):
        raise OracleCaseError("mode must be test|perf and backend a bare backend name")
    return [mode, "-b", backend, "--test-file", case_set.cases_path]


def check_witness(case_set: OracleCaseSet, output: str, *, expect_ran: bool = True) -> int:
    """Count witness hits; refuse when the fast path did not run (or ran under a kill switch)."""
    hits = len(re.findall(case_set.witness, output))
    if expect_ran and hits == 0:
        raise OracleCaseError(f"{case_set.op}/{case_set.name}: path witness never appeared; "
                              "the cases did not cover the candidate path")
    if not expect_ran and hits:
        raise OracleCaseError(f"{case_set.op}/{case_set.name}: path witness appeared {hits}x "
                              "under the kill switch")
    return hits
