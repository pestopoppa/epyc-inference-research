"""CPU_REPACK MUL_MAT_ID correctness gate for the DS41-1017 lane (and the shared repack GEMV).

WHY THIS EXISTS (operator 2026-10-21, "act on this advice"). The DS41 served-shape layer (a)
check (`served_shape_cases.py`, `gates.check_served_shape_case_set`) runs through
test-backend-ops, which allocates every weight in the plain CPU buffer. It never runs the
REPACKED MXFP4 kernel the 1017 server uses for every expert (`mxfp4_8x8_q8_0`: CPU_REPACK
buffer -> `repack.cpp` `forward_mul_mat_id` -> `ggml_gemv_mxfp4_8x8_q8_0` ->
`gemv_q4_b32_8x8_q8_0_lut_avx<block_mxfp4x8>`), which is why its MXFP4 NMSE reads exactly
0.0. All five DS41-1017 seeds (store inbox 30, S1-S5) change exactly that path, so a change
that breaks it would still pass layer (a). This module is the oracle that does run it.

WHY A STANDALONE PROBE, NOT test-backend-ops (option (b) of the advice). test-backend-ops'
eval mode copies the graph from backend 1 to backend 2 with `ggml_backend_graph_copy`, which
reads every tensor back (`get_tensor`). The CPU_REPACK buffer has `get_tensor = nullptr` and
stores a different layout, so a repacked weight can neither be copied nor compared that way;
its reference arm is also the generic CPU kernel on random uniform weights, not an
independent decode of the served experts. Making it repack-aware means restructuring the eval
loop of a frozen-lineage tool. The DS41-W5a harness (research 73aab47cc) already did the right
measurement; `cpu_repack_probe.cpp` generalises it (more types, a case file, engagement
witness, max-error metric, fail-closed markers) and is compiled at gate time against the
candidate's own ggml libraries, like `cpu_quant_reference_probe.cpp`.

WHAT IT MEASURES. Per case (`RepackCase`), the probe uploads the same weight bytes into the
CPU_REPACK buffer and a plain buffer, runs MUL_MAT_ID at every width in `WIDTHS`, and compares
both with a double-precision reference built from an independent scalar decode of the stored
bytes (cross-checked against ggml's to_float; any disagreement is `unavailable`). Two metrics:
`nmse` (sum sq err / sum sq ref) and `maxrel` (max |err| / rms(ref)). The second exists because
aggregate NMSE cannot see a localized fault: one wrong block scale on one row moves the NMSE by
~1e-6, below the anchor's own activation-quantization noise, while its max error is ~10x the
anchor's.

THE BOUNDS (`calibrate`). Measured on the ANCHOR build, per (case, width): nmse bound =
`served_shape_cases.tightened_nmse_bound(anchor_nmse, factor)` (anchor x factor, capped at
5e-4, floored at 1e-8 -- the served-shape formula, so the two layers agree); maxrel bound =
min(max(anchor x factor, MAXREL_FLOOR), MAXREL_CAP). An anchor at/above either cap refuses to
calibrate (investigate, do not clip). Recalibrate when the anchor moves (the manifest pins the
anchor's ggml library digests for the record, and the probe source digest, the case set, the
weight bytes and every per-width input hash for enforcement).

FAIL CLOSED (`check_case_set` returns `unavailable`, never a pass):
  * the lane has no repack case set, or the manifest is missing / malformed / for another lane,
    case set, probe source or case list;
  * the compiled probe lacks the case-set literals, or carries a TEST FIXTURE literal;
  * a case did not ENGAGE the repack path: `tensor->extra` unset (the repack did not claim the
    tensor, e.g. a type diverted by GGML_IQK), the repacked bytes equal the source bytes, or the
    engagement witness failed (perturbing one scale IN THE REPACKED BUFFER must move exactly
    output row 0 of expert 0, and restoring it must reproduce the clean output bit for bit);
  * weight bytes, inputs, the independent decode or the plain arm disagree with the manifest.
A measured violation of a bound is `wrong` (numerical).

SCOPE. `LANE_CASES` is lane-scoped: DS41 gets four real 1017 MXFP4 expert slices (8 experts of
4 tensors, served n_used = 6), the W5a set. `SHARED_CASES` cover the other two instantiations of
the shared GEMV template (`block_q4_0x8`, `block_iq4_nlx8`; the Q38FN lane serves IQ4_NL through
it) on served shapes with seeded synthetic weights; they run when a candidate touches a repack
source (`touches_repack`) and always at the bundle tier. Q4_0 is diverted from the repack when
GGML_IQK=1 (repack.cpp `ggml_repack_get_optimal_repack_type`), so its cases run with
GGML_IQK=0 -- the only configuration in which that instantiation executes at all.

TEST FIXTURE. `fixture_regression` compiles the probe with `-DAK_REPACK_FIXTURE=1|2`, which
leaves one scale perturbed in the repacked buffer during the measured runs -- a repack GEMV
that reads a wrong scale -- and proves the judge refuses it as `wrong` while the plain arm stays
clean. Fixture builds exist only in that function's temp dir; the gate refuses any binary that
carries the fixture literal, so it can never be the gate's binary.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
import datetime as _dt
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Callable, Literal, Mapping, Sequence

from . import served_shape_cases as ssc

CASE_SET_ID = "cpu_repack_mmid_v1"
PROBE_MARKER = "AK_CPU_REPACK_PROBE_V1"
PROBE_TYPES = "MXFP4,Q4_0,IQ4_NL"
#: Any literal starting with this marks a TEST FIXTURE build of the probe.
FIXTURE_LITERAL = "AK_REPACK_FIXTURE_"
FIXTURE_MODES = {1: "AK_REPACK_FIXTURE_LANE0_ALL_BLOCKS", 2: "AK_REPACK_FIXTURE_ONE_BLOCK_X4"}
MANIFEST_SCHEMA = "epyc.autokernel.cpu_repack_manifest.v1"
PROBE = Path(__file__).with_name("cpu_repack_probe.cpp")
#: Decode (1), speculative verify (2-5) and the prefill/batched shapes (8..512) a gemm-tile
#: change to the repack MMID walk (seed S5) would reach.
WIDTHS: tuple[int, ...] = (1, 2, 3, 4, 5, 8, 32, 128, 512)
DEFAULT_FACTOR = 3.0
NMSE_CAP = ssc.SERVED_SHAPE_NMSE_CAP
#: maxrel = max |y - ref| / rms(ref). Activation Q8_0 noise puts the anchor near 1e-2; a quarter
#: of the output RMS on any single element is never rounding.
MAXREL_CAP = 0.25
MAXREL_FLOOR = 1e-6
#: The plain (generic-path) arm is a sanity check on the reference, held to the generic bound.
PLAIN_NMSE_SANITY = NMSE_CAP
PROBE_TIMEOUT_S = 1800
COMPILE_TIMEOUT_S = 300
#: Routes whose window only this oracle reaches (`gates.PPL_CONTRACT_NO_OP_ORACLE`).
REPACK_ROUTES = frozenset({"cpu_repack_mmid", "cpu_repack_gemv"})
#: A candidate touching any of these runs the shared-template cases too.
REPACK_PATHS = frozenset({"ggml/src/ggml-cpu/repack.cpp", "ggml/src/ggml-cpu/repack.h",
                          "ggml/src/ggml-cpu/arch/x86/repack.cpp"})
#: The store layout: `<store>/cpu_repack/manifest.json`, next to `<store>/served_shape/`.
STORE_SUBDIR = "cpu_repack"

_DS41_DIR = "/mnt/raid0/llm/models/antirez/deepseek-v4.1-flash-gguf"
_DS41_SHARD = _DS41_DIR + "/DeepSeek-V4.1-Flash-1017-MXFP4-experts-{:05d}-of-00009.gguf"
#: The W5a expert slice: 8 experts spread over 0..383 (DS41 serves 6 of 384 per token).
_DS41_EXPERTS = (0, 47, 95, 143, 191, 239, 287, 383)


@dataclass(frozen=True)
class RepackCase:
    """One MUL_MAT_ID case of the probe (field order = the probe's TSV columns)."""
    name: str
    type_a: str          # MXFP4 | Q4_0 | IQ4_NL
    k: int               # ne[0], the reduction dim
    m: int               # ne[1], rows per expert
    n_as: int            # experts in the slice
    n_used: int          # experts routed per token
    bcast: bool          # gate/up: one activation row shared by the used experts
    src: str             # "gguf" (real served experts) | "synth" (seeded, quantized by ggml)
    path: str = "-"
    offset: int = 0      # absolute file offset of the tensor data (GGUFReader.data_offset)
    n_exp_total: int = 0
    expert_ids: tuple[int, ...] = ()
    seed: int = 0
    tensor: str = ""     # GGUF tensor name, re-verified at calibration
    #: GGML_IQK for the probe process: None = the served launch env (DS41/Q38FN: "1").
    ggml_iqk: "str | None" = None
    tier: str = "lane"   # "lane" | "shared"

    def __post_init__(self):
        if self.type_a not in PROBE_TYPES.split(","):
            raise ValueError(f"{self.name}: unsupported type {self.type_a}")
        if self.k % 32 or self.m % 8 or not (1 <= self.n_used <= self.n_as):
            raise ValueError(f"{self.name}: bad dims")
        if self.src == "gguf" and len(self.expert_ids) != self.n_as:
            raise ValueError(f"{self.name}: a gguf case names exactly n_as expert ids")
        if self.src not in ("gguf", "synth") or self.tier not in ("lane", "shared"):
            raise ValueError(f"{self.name}: bad src/tier")

    def tsv(self, widths: Sequence[int] = WIDTHS) -> str:
        return "\t".join((
            self.name, self.type_a, str(self.k), str(self.m), str(self.n_as), str(self.n_used),
            "1" if self.bcast else "0", ",".join(str(w) for w in widths), self.src,
            self.path if self.src == "gguf" else "-", str(self.offset if self.src == "gguf" else 0),
            str(self.n_exp_total), ",".join(str(e) for e in self.expert_ids) or "-", str(self.seed)))

    def identity(self) -> dict:
        body = asdict(self)
        body["expert_ids"] = list(self.expert_ids)
        return body


def _ds41(name, tensor, shard, offset, k, m, bcast):
    return RepackCase(name, "MXFP4", k, m, 8, 6, bcast, "gguf", _DS41_SHARD.format(shard),
                      offset, 384, _DS41_EXPERTS, tensor=tensor)


#: Lane-scoped real served-expert cases. DS41-1017: the W5a set (offsets from
#: `data/ds41-1017-w5-20261021/w5a/gguf_tensors.json`, GGUFReader absolute data offsets).
LANE_CASES: dict[str, tuple[RepackCase, ...]] = {
    "ds41": (
        _ds41("ds41_l3_gate_mxfp4", "blk.3.ffn_gate_exps.weight", 1, 24730394336, 5120, 2304, True),
        _ds41("ds41_l3_down_mxfp4", "blk.3.ffn_down_exps.weight", 1, 29568425696, 2304, 5120, False),
        _ds41("ds41_l20_up_mxfp4", "blk.20.ffn_up_exps.weight", 4, 4998643936, 5120, 2304, True),
        _ds41("ds41_l39_down_mxfp4", "blk.39.ffn_down_exps.weight", 7, 192, 2304, 5120, False),
    ),
}

#: The shared GEMV template's other two instantiations, on served shapes, synthetic weights.
SHARED_CASES: tuple[RepackCase, ...] = (
    RepackCase("shared_q4_0_ds41_gate_up", "Q4_0", 5120, 2304, 8, 6, True, "synth",
               seed=0x51345F30, ggml_iqk="0", tier="shared"),
    RepackCase("shared_q4_0_ds41_down", "Q4_0", 2304, 5120, 8, 6, False, "synth",
               seed=0x51345F31, ggml_iqk="0", tier="shared"),
    RepackCase("shared_iq4_nl_q38fn_down", "IQ4_NL", 640, 2560, 16, 10, False, "synth",
               seed=0x4951344E, tier="shared"),
    RepackCase("shared_iq4_nl_ds41_gate_up", "IQ4_NL", 5120, 2304, 8, 6, True, "synth",
               seed=0x4951344F, tier="shared"),
)


class ManifestRefused(ValueError):
    """The manifest cannot back a verdict (missing, malformed, stale or for another lane)."""


@dataclass(frozen=True)
class RepackResult:
    status: Literal["pass", "wrong", "unavailable"]
    reason: str = ""
    detail: str = ""
    records: tuple = ()


def lane_has_cases(lane: "str | None") -> bool:
    return lane in LANE_CASES


def manifest_path(store) -> Path:
    return Path(store) / STORE_SUBDIR / "manifest.json"


def touches_repack(paths) -> bool:
    return any(str(p) in REPACK_PATHS for p in paths or ())


def all_cases(lane: str) -> tuple[RepackCase, ...]:
    if lane not in LANE_CASES:
        raise ValueError(f"no CPU_REPACK case set for lane {lane!r}")
    return LANE_CASES[lane] + SHARED_CASES


def selected_cases(lane: str, include_shared: bool) -> tuple[RepackCase, ...]:
    return all_cases(lane) if include_shared else LANE_CASES[lane]


def probe_sha256(probe: Path = PROBE) -> str:
    return hashlib.sha256(Path(probe).read_bytes()).hexdigest()


# ------------------------------------------------------------------------------------ bounds
def maxrel_bound(anchor_maxrel: float, factor: float = DEFAULT_FACTOR) -> float:
    if not isinstance(anchor_maxrel, (int, float)) or isinstance(anchor_maxrel, bool) or \
            not math.isfinite(anchor_maxrel) or anchor_maxrel < 0:
        raise ValueError(f"anchor maxrel must be finite and >= 0, got {anchor_maxrel!r}")
    if anchor_maxrel >= MAXREL_CAP:
        raise ValueError(f"anchor maxrel {anchor_maxrel!r} is at/above the cap {MAXREL_CAP}: "
                         "the anchor itself is wrong here -- investigate, do not clip")
    return min(MAXREL_CAP, max(MAXREL_FLOOR, anchor_maxrel * factor))


# ------------------------------------------------------------------------------- probe output
def parse_output(text: str) -> dict:
    """{"headers": [...], "cases": {name: row}, "rows": {(name, n): row}, "ends": [...]}.
    Raises ValueError on malformed or duplicated rows."""
    out = {"headers": [], "cases": {}, "rows": {}, "ends": []}
    for lineno, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"probe output line {lineno} is not JSON: {exc}") from None
        kind = row.get("kind") if isinstance(row, dict) else None
        if kind == "header":
            out["headers"].append(row)
        elif kind == "end":
            out["ends"].append(row)
        elif kind == "case":
            if row.get("case") in out["cases"]:
                raise ValueError(f"case {row.get('case')!r} reported twice")
            out["cases"][row.get("case")] = row
        elif kind == "row":
            key = (row.get("case"), row.get("n"))
            if key in out["rows"]:
                raise ValueError(f"row {key!r} reported twice")
            out["rows"][key] = row
        else:
            raise ValueError(f"probe output line {lineno}: unknown kind {kind!r}")
    return out


def _merge(parsed: Sequence[dict]) -> dict:
    merged = {"headers": [], "cases": {}, "rows": {}, "ends": []}
    for part in parsed:
        merged["headers"] += part["headers"]
        merged["ends"] += part["ends"]
        for name, row in part["cases"].items():
            if name in merged["cases"]:
                raise ValueError(f"case {name!r} reported by two probe runs")
            merged["cases"][name] = row
        for key, row in part["rows"].items():
            if key in merged["rows"]:
                raise ValueError(f"row {key!r} reported by two probe runs")
            merged["rows"][key] = row
    return merged


def _header_refusal(headers, *, allow_fixture: bool = False) -> "str | None":
    if not headers:
        return "probe printed no header"
    for header in headers:
        if header.get("marker") != PROBE_MARKER or header.get("case_set") != CASE_SET_ID or \
                header.get("types") != PROBE_TYPES:
            return (f"probe header {header.get('marker')!r}/{header.get('case_set')!r}/"
                    f"{header.get('types')!r} is not {PROBE_MARKER}/{CASE_SET_ID}/{PROBE_TYPES}")
        if header.get("fixture") != "none" and not allow_fixture:
            return f"probe is a TEST FIXTURE build ({header.get('fixture')!r})"
        if header.get("repack_buft") is not True:
            return "this ggml build exposes no CPU_REPACK buffer type"
    return None


def engagement_refusal(case_row: Mapping) -> "str | None":
    """Why `case_row` does not prove the repack path ran, or None."""
    if case_row.get("repack_buft") != "CPU_REPACK":
        return f"weights not in CPU_REPACK (buffer {case_row.get('repack_buft')!r})"
    if case_row.get("extra") is not True:
        return ("tensor->extra unset: the repack did not claim this tensor "
                "(type/shape not repacked, or diverted by GGML_IQK)")
    if case_row.get("layout_changed") is not True:
        return "repacked bytes equal the source bytes (no repack happened)"
    if case_row.get("witness_unexpected") != 0:
        return (f"witness: perturbing one repacked scale moved {case_row.get('witness_unexpected')} "
                "outputs outside expert 0 / row 0 (layout or routing not what the probe assumes)")
    if not isinstance(case_row.get("witness_changed"), int) or case_row["witness_changed"] < 1:
        return "witness: perturbing a repacked scale changed no output (repacked bytes not read)"
    if case_row.get("witness_restored") is not True:
        return "witness: restoring the repacked scale did not reproduce the clean output"
    if case_row.get("engaged") is not True:
        return "probe reports the repack path not engaged"
    return None


def _num(row: Mapping, key: str) -> "float | None":
    value = row.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or \
            not math.isfinite(value) or value < 0:
        return None
    return float(value)


def judge(manifest: Mapping, parsed: Mapping, cases: Sequence[RepackCase], *,
          allow_fixture: bool = False) -> RepackResult:
    """Apply the manifest's per-(case, width) bounds to parsed probe output for `cases`."""
    refusal = _header_refusal(parsed.get("headers") or [], allow_fixture=allow_fixture)
    if refusal:
        return RepackResult("unavailable", refusal)
    if len(parsed.get("ends") or []) != len(parsed.get("headers") or []):
        return RepackResult("unavailable", "a probe run did not finish (no end row)")
    entries = {entry["case"]["name"]: entry for entry in manifest["cases"]}
    unavailable, wrong, records = [], [], []
    expected_rows = set()
    for case in cases:
        entry = entries.get(case.name)
        if entry is None:
            unavailable.append(f"{case.name}: not in the manifest")
            continue
        row = parsed["cases"].get(case.name)
        if row is None:
            unavailable.append(f"{case.name}: probe reported no such case")
            continue
        why = engagement_refusal(row)
        if why:
            unavailable.append(f"{case.name}: repack path did not engage: {why}")
            continue
        if row.get("raw_fnv64") != entry["raw_fnv64"]:
            unavailable.append(f"{case.name}: weight bytes {row.get('raw_fnv64')} != manifest "
                               f"{entry['raw_fnv64']} (model or quantizer changed; recalibrate)")
            continue
        if row.get("decode_maxdiff") != 0 or row.get("decode_inexact") != 0:
            unavailable.append(f"{case.name}: independent decoder disagrees with ggml to_float "
                               f"(maxdiff {row.get('decode_maxdiff')}, inexact "
                               f"{row.get('decode_inexact')})")
            continue
        for bound in entry["rows"]:
            n = bound["n"]
            expected_rows.add((case.name, n))
            r = parsed["rows"].get((case.name, n))
            if r is None:
                unavailable.append(f"{case.name} n={n}: no row")
                continue
            if r.get("input_fnv64") != bound["input_fnv64"]:
                unavailable.append(f"{case.name} n={n}: inputs {r.get('input_fnv64')} != "
                                   f"manifest {bound['input_fnv64']}")
                continue
            plain = _num(r, "nmse_plain")
            if r.get("finite_plain") is not True or plain is None or plain > PLAIN_NMSE_SANITY:
                unavailable.append(f"{case.name} n={n}: plain-arm sanity failed (nmse_plain "
                                   f"{r.get('nmse_plain')}); the reference is suspect")
                continue
            nmse, maxrel = _num(r, "nmse_repack"), _num(r, "maxrel_repack")
            record = {"case": case.name, "type": case.type_a, "n": n, "nmse": nmse,
                      "nmse_bound": bound["nmse_bound"], "maxrel": maxrel,
                      "maxrel_bound": bound["maxrel_bound"], "nmse_plain": plain}
            if r.get("finite_repack") is not True or nmse is None or maxrel is None:
                record["passed"] = False
                wrong.append(f"{case.name} n={n}: repack output non-finite")
            else:
                over = []
                if nmse > bound["nmse_bound"]:
                    over.append(f"nmse {nmse:.3e} > {bound['nmse_bound']:.3e}")
                if maxrel > bound["maxrel_bound"]:
                    over.append(f"maxrel {maxrel:.3e} > {bound['maxrel_bound']:.3e}")
                record["passed"] = not over
                if over:
                    wrong.append(f"{case.name}/{case.type_a} n={n}: " + ", ".join(over))
            records.append(record)
    extra = sorted(key for key in parsed["rows"]
                   if key not in expected_rows and key[0] in {c.name for c in cases})
    if extra:
        unavailable.append(f"rows not in the manifest: {extra[:4]}")
    detail = "\n".join(json.dumps(rec, sort_keys=True) for rec in records)
    if wrong:
        return RepackResult("wrong", f"{len(wrong)} repack row(s) out of bounds: "
                            + "; ".join(wrong[:4]), detail, tuple(records))
    if unavailable:
        return RepackResult("unavailable", f"{len(unavailable)} problem(s): "
                            + "; ".join(unavailable[:4]), detail, tuple(records))
    if not records:
        return RepackResult("unavailable", "no repack row was judged (empty case set)")
    return RepackResult("pass", f"{len(records)} repack row(s) over {len(cases)} case(s) "
                        "within anchor bounds, repack engaged on every case", detail,
                        tuple(records))


# ------------------------------------------------------------------------------------ manifest
def load_manifest(path, *, lane: str, probe: Path = PROBE) -> dict:
    path = Path(path)
    try:
        body = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ManifestRefused(f"no CPU_REPACK manifest at {path} (calibrate on the anchor: "
                              "python -m autokernel.loop.cpu_repack_cases calibrate)") from None
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifestRefused(f"CPU_REPACK manifest {path} unreadable: {exc}") from None
    if not isinstance(body, dict) or body.get("schema") != MANIFEST_SCHEMA:
        raise ManifestRefused(f"{path}: schema is not {MANIFEST_SCHEMA}")
    if body.get("case_set") != CASE_SET_ID or body.get("lane") != lane:
        raise ManifestRefused(f"{path}: case set/lane {body.get('case_set')!r}/"
                              f"{body.get('lane')!r} != {CASE_SET_ID}/{lane}")
    if body.get("probe_sha256") != probe_sha256(probe):
        raise ManifestRefused(f"{path}: calibrated with another probe source "
                              f"({str(body.get('probe_sha256'))[:12]}); recalibrate")
    try:
        want = [case.identity() for case in all_cases(lane)]
        have = [entry["case"] for entry in body["cases"]]
        for entry in body["cases"]:
            if [row["n"] for row in entry["rows"]] != list(WIDTHS):
                raise ManifestRefused(f"{path}: {entry['case']['name']} widths != {WIDTHS}")
            for row in entry["rows"]:
                for key in ("nmse_bound", "maxrel_bound"):
                    value = row[key]
                    if isinstance(value, bool) or not isinstance(value, (int, float)) or \
                            not math.isfinite(value) or value <= 0:
                        raise ManifestRefused(f"{path}: bad {key} {value!r}")
                if not isinstance(row["input_fnv64"], str):
                    raise ManifestRefused(f"{path}: bad input hash")
            if not isinstance(entry["raw_fnv64"], str):
                raise ManifestRefused(f"{path}: bad weight hash")
    except (KeyError, TypeError) as exc:
        raise ManifestRefused(f"{path}: malformed ({type(exc).__name__}: {exc})") from None
    if have != want:
        raise ManifestRefused(f"{path}: case list differs from this code's {CASE_SET_ID} "
                              f"set for lane {lane}; recalibrate")
    return body


# ----------------------------------------------------------------------------- build and run
def _cmake_cache(build_dir: Path) -> dict:
    values = {}
    try:
        for line in (Path(build_dir) / "CMakeCache.txt").read_text(encoding="utf-8").splitlines():
            if ":" in line and "=" in line and not line.startswith(("#", "//")):
                key, _, value = line.partition("=")
                values[key.split(":", 1)[0]] = value
    except OSError:
        pass
    return values


def build_inputs(build_dir: Path) -> tuple["Path | None", str, str]:
    """(ggml source root, C++ compiler, refusal-or-empty) for a build dir."""
    cache = _cmake_cache(build_dir)
    root = cache.get("CMAKE_HOME_DIRECTORY")
    if not root or not (Path(root) / "ggml/include/ggml.h").is_file():
        return None, "", f"{build_dir}: CMakeCache.txt names no source tree with ggml headers"
    compiler = cache.get("CMAKE_CXX_COMPILER") or "c++"
    return Path(root), compiler, ""


def compile_probe(build_dir: Path, out: Path, *, defines: Sequence[str] = (),
                  probe: Path = PROBE) -> tuple["Path | None", str]:
    build_dir = Path(build_dir)
    lib = build_dir / "bin"
    root, compiler, refusal = build_inputs(build_dir)
    if refusal:
        return None, refusal
    missing = [str(p) for p in (lib / "libggml.so", lib / "libggml-base.so",
                                lib / "libggml-cpu.so", probe) if not p.is_file()]
    if missing:
        return None, "missing: " + ", ".join(missing)
    command = [compiler, "-std=c++17", "-O2", "-march=native", "-fopenmp",
               *(f"-D{d}" for d in defines), "-I", str(root / "ggml/include"), str(probe),
               "-L", str(lib), "-Wl,-rpath," + str(lib), "-lggml-cpu", "-lggml-base", "-lggml",
               "-o", str(out)]
    # The loop's own toolchain env (a candidate launch env has no PATH; cpu_quant_reference).
    done = subprocess.run(command, capture_output=True, text=True, timeout=COMPILE_TIMEOUT_S,
                          env=dict(os.environ))
    if done.returncode:
        return None, "probe compile failed: " + done.stderr[-2000:]
    return Path(out), ""


def binary_refusal(binary: Path) -> "str | None":
    """The compiled probe must carry the case-set literals and no TEST FIXTURE literal."""
    try:
        data = Path(binary).read_bytes()
    except OSError as exc:
        return f"probe binary unreadable: {exc}"
    for literal in (PROBE_MARKER, CASE_SET_ID, PROBE_TYPES):
        if literal.encode() not in data:
            return f"probe binary lacks the {literal!r} literal (not the {CASE_SET_ID} probe)"
    if FIXTURE_LITERAL.encode() in data:
        return "probe binary is a TEST FIXTURE build (carries the fixture literal); refused"
    return None


def _probe_env(build_dir: Path, launch_env: "Mapping[str, str] | None", ggml_iqk) -> dict:
    env = dict(os.environ if launch_env is None else launch_env)
    lib = str(Path(build_dir) / "bin")
    env["LD_LIBRARY_PATH"] = lib + (":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
    if ggml_iqk is not None:
        env["GGML_IQK"] = ggml_iqk
    return env


def run_probe(binary: Path, build_dir: Path, cases: Sequence[RepackCase], *, threads: int,
              launch_env: "Mapping[str, str] | None" = None,
              topology_prefix: Sequence[str] = (), workdir: Path) -> dict:
    """Run every case (one probe process per GGML_IQK group); merged parsed output.
    Raises RuntimeError on a probe failure, ValueError on malformed output."""
    groups: dict = {}
    for case in cases:
        groups.setdefault(case.ggml_iqk, []).append(case)
    parsed = []
    for index, (iqk, group) in enumerate(sorted(groups.items(), key=lambda kv: str(kv[0]))):
        tsv, out = Path(workdir) / f"cases-{index}.tsv", Path(workdir) / f"out-{index}.jsonl"
        tsv.write_text("".join(case.tsv() + "\n" for case in group), encoding="utf-8")
        done = subprocess.run([*topology_prefix, str(binary), str(threads), str(tsv), str(out)],
                              capture_output=True, text=True, timeout=PROBE_TIMEOUT_S,
                              env=_probe_env(build_dir, launch_env, iqk))
        if done.returncode:
            raise RuntimeError(f"probe (GGML_IQK={iqk}) exit {done.returncode}: "
                               f"{done.stderr[-1500:]}")
        parsed.append(parse_output(out.read_text(encoding="utf-8")))
    return _merge(parsed)


def check_case_set(build_dir, manifest_path, *, lane: str, include_shared: bool, threads: int,
                   launch_env: "Mapping[str, str] | None" = None,
                   topology_prefix: Sequence[str] = (),
                   _compile: "Callable | None" = None, _run: "Callable | None" = None
                   ) -> RepackResult:
    """THE GATE. Fails closed on every missing piece (see module docstring)."""
    if not lane_has_cases(lane):
        return RepackResult("unavailable", f"no CPU_REPACK case set for lane {lane!r}")
    try:
        manifest = load_manifest(manifest_path, lane=lane)
    except ManifestRefused as exc:
        return RepackResult("unavailable", f"CPU_REPACK manifest refused: {exc}")
    cases = selected_cases(lane, include_shared)
    try:
        with tempfile.TemporaryDirectory(prefix="ak-cpu-repack-") as temp:
            binary, why = (_compile or compile_probe)(Path(build_dir), Path(temp) / "cpu-repack-probe")
            if binary is None:
                return RepackResult("unavailable", f"CPU_REPACK probe unavailable: {why}")
            why = binary_refusal(binary)
            if why:
                return RepackResult("unavailable", why)
            parsed = (_run or run_probe)(binary, Path(build_dir), cases, threads=threads,
                                         launch_env=launch_env, topology_prefix=topology_prefix,
                                         workdir=Path(temp))
    except (OSError, RuntimeError, ValueError, subprocess.TimeoutExpired) as exc:
        return RepackResult("unavailable", f"CPU_REPACK probe failed: {type(exc).__name__}: {exc}")
    return judge(manifest, parsed, cases)


# --------------------------------------------------------------------------------- calibration
def verify_gguf_offsets(cases: Sequence[RepackCase], *,
                        gguf_py: str = "/mnt/raid0/llm/llama.cpp/gguf-py") -> None:
    """Header-only check that each gguf case's tensor sits at its offset with its type/shape."""
    if gguf_py not in sys.path:
        sys.path.insert(0, gguf_py)
    from gguf import GGUFReader  # noqa: PLC0415
    readers: dict = {}
    for case in cases:
        if case.src != "gguf":
            continue
        reader = readers.setdefault(case.path, GGUFReader(case.path))
        tensor = next((t for t in reader.tensors if t.name == case.tensor), None)
        if tensor is None:
            raise ValueError(f"{case.name}: {case.tensor} not in {case.path}")
        shape = [int(x) for x in tensor.shape]
        if int(tensor.data_offset) != case.offset or tensor.tensor_type.name != case.type_a or \
                shape != [case.k, case.m, case.n_exp_total]:
            raise ValueError(f"{case.name}: GGUF says offset {int(tensor.data_offset)} type "
                             f"{tensor.tensor_type.name} shape {shape}; the case says "
                             f"{case.offset} {case.type_a} {[case.k, case.m, case.n_exp_total]}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_from_measurement(parsed: Mapping, *, lane: str, factor: float = DEFAULT_FACTOR,
                              anchor: Mapping | None = None, threads: int = 0) -> dict:
    """Bounds from an ANCHOR measurement. Raises ValueError on anything that cannot back a
    bound: a header/engagement/decode problem, a missing row, an anchor at/above a cap."""
    refusal = _header_refusal(parsed.get("headers") or [])
    if refusal:
        raise ValueError(f"anchor probe output refused: {refusal}")
    entries = []
    for case in all_cases(lane):
        row = parsed["cases"].get(case.name)
        if row is None:
            raise ValueError(f"anchor probe reported no {case.name}")
        why = engagement_refusal(row)
        if why:
            raise ValueError(f"anchor: {case.name}: repack path did not engage: {why}")
        if row.get("decode_maxdiff") != 0 or row.get("decode_inexact") != 0:
            raise ValueError(f"anchor: {case.name}: independent decode disagrees with ggml")
        bounds = []
        for n in WIDTHS:
            r = parsed["rows"].get((case.name, n))
            if r is None:
                raise ValueError(f"anchor: {case.name} n={n}: no row")
            nmse, maxrel, plain = (_num(r, "nmse_repack"), _num(r, "maxrel_repack"),
                                   _num(r, "nmse_plain"))
            if r.get("finite_repack") is not True or nmse is None or maxrel is None:
                raise ValueError(f"anchor: {case.name} n={n}: non-finite repack output")
            if r.get("finite_plain") is not True or plain is None or plain > PLAIN_NMSE_SANITY:
                raise ValueError(f"anchor: {case.name} n={n}: plain-arm sanity failed ({plain})")
            bounds.append({"n": n, "input_fnv64": r["input_fnv64"], "anchor_nmse": nmse,
                           "anchor_maxrel": maxrel, "anchor_nmse_plain": plain,
                           "nmse_bound": ssc.tightened_nmse_bound(nmse, factor=factor),
                           "maxrel_bound": maxrel_bound(maxrel, factor)})
        entries.append({"case": case.identity(), "raw_fnv64": row["raw_fnv64"], "rows": bounds})
    return {"schema": MANIFEST_SCHEMA, "case_set": CASE_SET_ID, "lane": lane,
            "probe_sha256": probe_sha256(), "factor": factor, "widths": list(WIDTHS),
            "nmse_cap": NMSE_CAP, "maxrel_cap": MAXREL_CAP, "threads": threads,
            "anchor": dict(anchor or {}),
            "calibrated_at": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "cases": entries}


def calibrate(build_dir, out: Path, *, lane: str, threads: int, factor: float = DEFAULT_FACTOR,
              launch_env=None, topology_prefix: Sequence[str] = (), verify_gguf: bool = True,
              raw_out: "Path | None" = None) -> dict:
    build_dir = Path(build_dir)
    cases = all_cases(lane)
    if verify_gguf:
        verify_gguf_offsets(cases)
    with tempfile.TemporaryDirectory(prefix="ak-cpu-repack-cal-") as temp:
        binary, why = compile_probe(build_dir, Path(temp) / "cpu-repack-probe")
        if binary is None:
            raise RuntimeError(why)
        why = binary_refusal(binary)
        if why:
            raise RuntimeError(why)
        parsed = run_probe(binary, build_dir, cases, threads=threads, launch_env=launch_env,
                           topology_prefix=topology_prefix, workdir=Path(temp))
        if raw_out is not None:
            raw_out.write_text("".join((Path(temp) / name).read_text(encoding="utf-8")
                                       for name in sorted(os.listdir(temp))
                                       if name.startswith("out-")), encoding="utf-8")
    lib = build_dir / "bin"
    anchor = {"build_dir": str(build_dir),
              "libs": {name: _sha256((lib / name).resolve())
                       for name in ("libggml-cpu.so", "libggml-base.so", "libggml.so")}}
    manifest = manifest_from_measurement(parsed, lane=lane, factor=factor, anchor=anchor,
                                         threads=threads)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(f".{out.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, out)
    return manifest


# ------------------------------------------------------------------------- broken-kernel test
def fixture_regression(build_dir, manifest_path, *, lane: str, threads: int, launch_env=None,
                       topology_prefix: Sequence[str] = ()) -> dict:
    """Prove the gate catches a repack GEMV that reads a wrong scale, on real hardware.

    For each fixture mode: the gate's binary check refuses the fixture build outright, and the
    judge (fixture header allowed, everything else unchanged) returns `wrong`, with the plain
    arm clean. Also proves the engagement assertion fires for real: the Q4_0 shared cases run
    with GGML_IQK=1 (the repack does not claim Q4_0 then) must judge `unavailable` as
    "did not engage". Returns a report; `report["passed"]` is the verdict."""
    build_dir = Path(build_dir)
    manifest = load_manifest(manifest_path, lane=lane)
    cases = all_cases(lane)
    report = {"lane": lane, "modes": {}, "engagement_negative": None}
    passed = True
    with tempfile.TemporaryDirectory(prefix="ak-cpu-repack-fixture-") as temp:
        for mode, literal in FIXTURE_MODES.items():
            work = Path(temp) / f"mode{mode}"
            work.mkdir()
            binary, why = compile_probe(build_dir, work / "probe-fixture",
                                        defines=(f"AK_REPACK_FIXTURE={mode}",))
            if binary is None:
                raise RuntimeError(why)
            refused = binary_refusal(binary)
            parsed = run_probe(binary, build_dir, cases, threads=threads, launch_env=launch_env,
                               topology_prefix=topology_prefix, workdir=work)
            result = judge(manifest, parsed, cases, allow_fixture=True)
            plain_clean = all(rec["nmse_plain"] is not None and
                              rec["nmse_plain"] <= PLAIN_NMSE_SANITY for rec in result.records)
            failing = [rec for rec in result.records if not rec["passed"]]
            ok = (refused is not None and "FIXTURE" in refused
                  and any(h.get("fixture") == literal for h in parsed["headers"])
                  and result.status == "wrong" and plain_clean
                  and {rec["case"] for rec in failing} == {c.name for c in cases})
            report["modes"][literal] = {
                "binary_refused_by_gate": refused, "judge_status": result.status,
                "judge_reason": result.reason[:600], "failing_rows": len(failing),
                "rows": len(result.records), "failing_cases": sorted({r["case"] for r in failing}),
                "caught_by_nmse": sum(1 for r in failing if r["nmse"] is not None
                                      and r["nmse"] > r["nmse_bound"]),
                "caught_by_maxrel": sum(1 for r in failing if r["maxrel"] is not None
                                        and r["maxrel"] > r["maxrel_bound"]),
                "plain_clean": plain_clean, "passed": ok}
            passed = passed and ok
        # engagement negative: Q4_0 under GGML_IQK=1 is never repacked
        work = Path(temp) / "negative"
        work.mkdir()
        binary, why = compile_probe(build_dir, work / "probe")
        if binary is None:
            raise RuntimeError(why)
        q4 = tuple(RepackCase(**{**c.identity(), "expert_ids": tuple(c.expert_ids),
                                 "ggml_iqk": "1"}) for c in SHARED_CASES if c.type_a == "Q4_0")
        parsed = run_probe(binary, build_dir, q4, threads=threads, launch_env=launch_env,
                           topology_prefix=topology_prefix, workdir=work)
        by_name = {c.name: c for c in q4}
        result = judge(manifest, parsed, tuple(by_name.values()))
        ok = result.status == "unavailable" and "did not engage" in result.reason and all(
            parsed["cases"][name].get("extra") is False for name in by_name)
        report["engagement_negative"] = {"judge_status": result.status,
                                         "judge_reason": result.reason[:600], "passed": ok}
        passed = passed and ok
    report["passed"] = passed
    return report


# ------------------------------------------------------------------------------------------ CLI
def _launch_env_from_os() -> dict:
    return dict(os.environ)


def main(argv: "Sequence[str] | None" = None) -> int:
    parser = argparse.ArgumentParser(prog="cpu_repack_cases", description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("calibrate", "check", "fixture-regression"):
        p = sub.add_parser(name)
        p.add_argument("--build", required=True, type=Path)
        p.add_argument("--lane", default="ds41")
        p.add_argument("--threads", required=True, type=int)
        p.add_argument("--manifest", required=True, type=Path)
        p.add_argument("--report", type=Path)
        if name == "calibrate":
            p.add_argument("--factor", type=float, default=DEFAULT_FACTOR)
            p.add_argument("--raw-out", type=Path)
        if name == "check":
            p.add_argument("--shared", action="store_true")
    args = parser.parse_args(argv)
    env = _launch_env_from_os()
    if args.cmd == "calibrate":
        manifest = calibrate(args.build, args.manifest, lane=args.lane, threads=args.threads,
                             factor=args.factor, launch_env=env, raw_out=args.raw_out)
        body = {"manifest": str(args.manifest), "cases": len(manifest["cases"]),
                "rows": sum(len(e["rows"]) for e in manifest["cases"]),
                "max_anchor_nmse": max(r["anchor_nmse"] for e in manifest["cases"] for r in e["rows"]),
                "max_anchor_maxrel": max(r["anchor_maxrel"] for e in manifest["cases"] for r in e["rows"])}
        rc = 0
    elif args.cmd == "check":
        result = check_case_set(args.build, args.manifest, lane=args.lane,
                                include_shared=args.shared, threads=args.threads, launch_env=env)
        body = {"status": result.status, "reason": result.reason, "rows": len(result.records)}
        rc = 0 if result.status == "pass" else 1
    else:
        body = fixture_regression(args.build, args.manifest, lane=args.lane,
                                  threads=args.threads, launch_env=env)
        rc = 0 if body["passed"] else 1
    text = json.dumps(body, indent=1, sort_keys=True)
    if args.report:
        args.report.write_text(text + "\n", encoding="utf-8")
    print(text)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["CASE_SET_ID", "LANE_CASES", "ManifestRefused", "REPACK_PATHS", "REPACK_ROUTES",
           "RepackCase", "RepackResult", "SHARED_CASES", "WIDTHS", "binary_refusal",
           "calibrate", "check_case_set", "engagement_refusal", "fixture_regression", "judge",
           "lane_has_cases", "load_manifest", "manifest_from_measurement", "manifest_path",
           "maxrel_bound", "parse_output", "selected_cases", "touches_repack"]
