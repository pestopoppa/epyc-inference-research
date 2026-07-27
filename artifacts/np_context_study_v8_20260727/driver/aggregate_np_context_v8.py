#!/usr/bin/env python3
"""Read-only status and throughput aggregation for the v8 TB-6 surface.

The runner intentionally allows a cell to finish as either a measured
``results.json`` or an explicit capacity ``skip.txt``.  This helper treats a
surface as publishable only after its driver wrote ``complete.txt`` and every
required cell has one valid terminal disposition.  It never starts a server,
modifies a run directory, or writes an aggregate unless explicitly asked to do
so and all requested surfaces are terminal.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shlex
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


FULL_GRID = {
    2048: (1, 2, 4, 8, 16, 32),
    8192: (1, 2, 4, 8, 16, 32),
    16384: (1, 2, 4, 8, 16, 32),
    32768: (1, 2, 4, 8, 16, 32),
}
A4_BRIDGE_GRID = {2048: (1, 2, 4, 8, 16, 32), 8192: (1, 2, 4, 8, 16)}
GRID_CONTRACTS = {"full": "cartesian24_v2", "a4_bridge": "a4_bridge11_v1"}
PROVENANCE_RE = re.compile(r"^mode=(?P<mode>\S+) grid=(?P<grid>\S+)\b", re.MULTILINE)
MTP_DEPTH_RE = re.compile(r"\bmtp_depth=(?P<mtp_depth>\d+)\b")
GRID_CONTRACT_LINE_RE = re.compile(r"^grid_contract=(?P<contract>\S+)$", re.MULTILINE)
GRID_CONTRACT_ANY_RE = re.compile(r"^grid_contract=.*$", re.MULTILINE)
RESOURCE_SKIP_RE = re.compile(r"^SKIP n_ctx_slot=(?P<nctx>\d+) vram=(?P<vram>\d+)G requested_L=(?P<length>\d+)\n?$")
CAPACITY_START_SKIP_RE = re.compile(
    r"^SKIP capacity_start signature=(?P<signature>[a-z0-9_]+) "
    r"stderr_sha256=(?P<stderr_sha256>[0-9a-f]{64}) "
    r"server_argv_sha256=(?P<argv_sha256>[0-9a-f]{64}) requested_np=(?P<np>\d+) "
    r"requested_L=(?P<length>\d+) requested_ctx=(?P<context>\d+)\n?$"
)
CAPACITY_START_SIGNATURES = {
    "hip_error_out_of_memory": re.compile(r"\bhipErrorOutOfMemory\b", re.IGNORECASE),
    "allocation_failure": re.compile(
        r"failed to allocate .*\b(HIP|ROCm|VRAM|GPU|device|KV|buffer)\b", re.IGNORECASE
    ),
    "rocm_memory_allocation_failure": re.compile(
        r"\b(HIP|ROCm)\b.*\b(out of memory|memory allocation)\b", re.IGNORECASE
    ),
}
COMPLETE_RE = re.compile(r"^COMPLETE \d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z\n$")
CANONICAL_LABELS = (
    "A3_tc_thinkingcap_q8",
    "A3_ff_fable_non_mtp_q8",
    "A3_ff_fable_mtp_q8",
    "Laguna_ud_iq2_gpu_dflash_off",
    "A4_35b_a3b_v8_bridge",
)
CANONICAL_THROUGHPUT_PROMPTS = (
    "/mnt/raid0/llm/epyc-inference-research/"
    "artifacts/architect-bench-gpu-20260720/questions_olympiadbench_hard.json"
)
V8_BINARY = "/mnt/raid0/llm/llama.cpp/build-hip/bin/llama-server"
CANONICAL_SURFACE_BINDINGS = {
    "A3_tc_thinkingcap_q8": ("full", "full", "/mnt/raid0/llm/models/ThinkingCap-Qwen3.6-27B-GGUF/ThinkingCap-Qwen3.6-27B-Q8_0.gguf"),
    "A3_ff_fable_non_mtp_q8": ("full", "full", "/mnt/raid0/llm/models/Qwen3.6-27B-Fable-Fusion-711-GGUF/Qwen3.6-27B-Fable-Fus-711-UnHeretic-NM-DAU-NEO-MAX-NEO-Q8_0.gguf"),
    "A3_ff_fable_mtp_q8": ("full", "full", "/mnt/raid0/llm/models/Qwen3.6-27B-Fable-Fusion-711-GGUF/Qwen3.6-27B-Fable-Fus-711-UnHeretic-NM-DAU-NEO-MAX-NEO-MTP-Q8_0.gguf"),
    "Laguna_ud_iq2_gpu_dflash_off": ("throughput_only", "full", "/mnt/raid0/llm/models/Laguna-S-2.1-GGUF/Laguna-S-2.1-UD-IQ2_M.gguf"),
    "A4_35b_a3b_v8_bridge": ("throughput_only", "a4_bridge", "/mnt/raid0/llm/models/Qwen3.6-35B-A3B-MTP-Q8_0.gguf"),
}
CANONICAL_MTP_DEPTHS = {
    "A3_tc_thinkingcap_q8": 4,
    "A3_ff_fable_non_mtp_q8": 0,
    "A3_ff_fable_mtp_q8": 1,
    "Laguna_ud_iq2_gpu_dflash_off": 0,
    "A4_35b_a3b_v8_bridge": 4,
}


def validate_quality(
    directory: Path,
    *,
    label: str,
    model: str,
    suite_name: str,
    expected_n: int,
    max_tokens: int,
    questions: str,
    thinking: bool,
) -> str | None:
    """Validate a sealed rb1024 capture without re-running inference.

    This deliberately checks both the summary and every saved response.  A
    zero-error summary alone is not a terminal capture: it could be a partial
    or stale file left by an interrupted runner.
    """
    summary_path = directory / "summary.json"
    rows_path = directory / "per_question.jsonl"
    try:
        payload = json.loads(summary_path.read_text())
        rows = [json.loads(line) for line in rows_path.read_text().splitlines() if line.strip()]
    except (OSError, json.JSONDecodeError) as exc:
        return f"unreadable quality capture: {exc}"
    meta, suites = payload.get("meta"), payload.get("suites")
    expected_meta = {
        "kernel": "production-consolidated-v8",
        "arm": f"{label}_rb1024_{suite_name}",
        "models": model,
        "questions_pinned": questions,
        "max_tokens": max_tokens,
        "enable_thinking": thinking,
        "endpoint": "chat",
        "repeats": 1,
        "n_per_suite": expected_n,
    }
    if not isinstance(meta, dict):
        return "quality summary has no meta object"
    for key, expected in expected_meta.items():
        if meta.get(key) != expected:
            return f"quality meta.{key} is not {expected!r}"
    if not isinstance(suites, list) or len(suites) != 1 or not isinstance(suites[0], dict):
        return "quality summary does not contain exactly one suite"
    suite = suites[0]
    if suite.get("suite") != suite_name:
        return f"quality suite is not {suite_name}"
    if suite.get("n") != expected_n or suite.get("n_questions") != expected_n:
        return f"quality n/n_questions are not {expected_n}"
    if suite.get("errors", 0) != 0:
        return f"quality runner recorded {suite['errors']} request error(s)"
    if len(rows) != expected_n:
        return f"quality per_question rows are {len(rows)}, expected {expected_n}"
    if any(not isinstance(row, dict) for row in rows):
        return "quality per_question contains a non-object row"
    # The runner records one draw per supplied pinned item.  Repeated ids would
    # make a partial resume look complete even when the line count is right.
    ids = [row.get("question_id") or row.get("instance_id") or row.get("id") for row in rows]
    if any(value is None for value in ids) or len(set(ids)) != expected_n:
        return "quality per_question ids are missing or not unique"
    if any(row.get("request_error") for row in rows):
        return "quality per_question contains a request error"
    return None


@dataclass(frozen=True)
class SurfaceSpec:
    label: str
    mode: str
    grid: str
    mtp_depth: int
    cells: tuple[tuple[int, int], ...]


def required_cells(grid: str) -> tuple[tuple[int, int], ...]:
    if grid == "full":
        source = FULL_GRID
    elif grid == "a4_bridge":
        source = A4_BRIDGE_GRID
    else:
        raise ValueError(f"unsupported grid {grid!r}")
    return tuple((np, length) for length, nps in source.items() for np in nps)


def load_spec(base: Path) -> SurfaceSpec:
    provenance = base / "provenance.txt"
    if not provenance.is_file():
        raise ValueError("missing provenance.txt")
    provenance_text = provenance.read_text()
    match = PROVENANCE_RE.search(provenance_text)
    if not match:
        raise ValueError("provenance has no mode/grid declaration")
    mode, grid = match.group("mode"), match.group("grid")
    if mode not in {"full", "throughput_only"}:
        raise ValueError(f"unsupported mode {mode!r}")
    contract_lines = GRID_CONTRACT_ANY_RE.findall(provenance_text)
    if len(contract_lines) != 1:
        raise ValueError("provenance must contain exactly one grid_contract line")
    contract_match = GRID_CONTRACT_LINE_RE.fullmatch(contract_lines[0])
    expected_contract = GRID_CONTRACTS.get(grid)
    if contract_match is None or contract_match.group("contract") != expected_contract:
        raise ValueError(f"grid_contract is not canonical {expected_contract!r}")
    binding = CANONICAL_SURFACE_BINDINGS.get(base.name)
    if binding and (mode, grid) != binding[:2]:
        raise ValueError(f"canonical {base.name} requires mode/grid {binding[:2]!r}")
    mtp_match = MTP_DEPTH_RE.search(provenance_text)
    if not mtp_match:
        raise ValueError("provenance has no mtp_depth declaration")
    mtp_depth = int(mtp_match.group("mtp_depth"))
    expected_mtp_depth = CANONICAL_MTP_DEPTHS.get(base.name)
    if expected_mtp_depth is not None and mtp_depth != expected_mtp_depth:
        raise ValueError(f"canonical {base.name} requires mtp_depth {expected_mtp_depth}")
    return SurfaceSpec(base.name, mode, grid, mtp_depth, required_cells(grid))


def finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def validate_result(path: Path, label: str, np: int, length: int) -> tuple[dict[str, Any] | None, str | None]:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return None, f"invalid results.json: {exc}"
    meta, suites = payload.get("meta"), payload.get("suites")
    expected_arm = f"{label}_np{np}_L{length}"
    required_meta = {
        "kernel": "production-consolidated-v8",
        "arm": expected_arm,
        "max_tokens": length,
        "questions_pinned": CANONICAL_THROUGHPUT_PROMPTS,
        "enable_thinking": False,
    }
    if not isinstance(meta, dict):
        return None, "missing meta object"
    for key, expected in required_meta.items():
        if meta.get(key) != expected:
            return None, f"meta.{key} is not {expected!r}"
    binding = CANONICAL_SURFACE_BINDINGS.get(label)
    if binding:
        if meta.get("models") != binding[2]:
            return None, f"meta.models is not canonical model {binding[2]!r}"
        if meta.get("binary") != V8_BINARY:
            return None, f"meta.binary is not frozen v8 binary {V8_BINARY!r}"
    if not isinstance(suites, list) or len(suites) != 1 or not isinstance(suites[0], dict):
        return None, "expected exactly one suite result"
    suite = suites[0]
    if suite.get("suite") != "olympiadbench_hard":
        return None, "suite is not olympiadbench_hard"
    if suite.get("n") != np or suite.get("n_questions") != np:
        return None, f"suite n/n_questions are not {np}"
    throughput = suite.get("throughput")
    if not isinstance(throughput, dict) or throughput.get("concurrency") != np:
        return None, f"throughput.concurrency is not {np}"
    required = ("wall_s", "completion_tokens", "prompt_tokens", "aggregate_decode_tok_s", "aggregate_total_tok_s")
    if any(not finite_number(throughput.get(key)) for key in required):
        return None, "throughput has missing or non-finite numeric field"
    if any(throughput[key] <= 0 for key in required):
        return None, "throughput fields must be positive"
    if suite.get("errors", 0) != 0:
        return None, f"runner recorded {suite['errors']} request error(s)"
    return throughput, None


def validate_skip(path: Path, label: str, mtp_depth: int, np: int, length: int) -> tuple[str | None, str | None]:
    try:
        text = path.read_text()
    except OSError as exc:
        return None, f"unreadable skip.txt: {exc}"
    resource_match = RESOURCE_SKIP_RE.fullmatch(text)
    if resource_match:
        nctx, vram, requested = (int(resource_match.group(name)) for name in ("nctx", "vram", "length"))
        if requested != length:
            return None, f"skip requested_L is not {length}"
        if nctx >= length and vram <= 61:
            return None, "skip has neither insufficient slot context nor VRAM overflow"
        return text.strip(), None

    capacity_match = CAPACITY_START_SKIP_RE.fullmatch(text)
    if not capacity_match:
        return None, "skip.txt does not match a canonical resource or capacity-start skip grammar"
    requested_np, requested_length, requested_context = (
        int(capacity_match.group(name)) for name in ("np", "length", "context")
    )
    if (requested_np, requested_length) != (np, length):
        return None, "capacity-start skip request does not bind this cell"
    if requested_context != np * length:
        return None, "capacity-start skip requested_ctx is not requested_np * requested_L"
    if requested_context < 262144:
        return None, "capacity-start skip is only valid for a newly scheduled >=262144 context cell"
    signature = capacity_match.group("signature")
    pattern = CAPACITY_START_SIGNATURES.get(signature)
    if pattern is None:
        return None, "capacity-start skip has an unrecognized allocation signature"
    stderr = path.with_name("server.stderr")
    try:
        stderr_bytes = stderr.read_bytes()
    except OSError as exc:
        return None, f"capacity-start skip has unreadable server.stderr: {exc}"
    actual_hash = hashlib.sha256(stderr_bytes).hexdigest()
    if actual_hash != capacity_match.group("stderr_sha256"):
        return None, "capacity-start skip server.stderr SHA-256 mismatch"
    if not pattern.search(stderr_bytes.decode("utf-8", errors="replace")):
        return None, "capacity-start skip signature is absent from server.stderr"
    argv = path.with_name("server.argv")
    try:
        argv_bytes = argv.read_bytes()
    except OSError as exc:
        return None, f"capacity-start skip has unreadable server.argv: {exc}"
    if hashlib.sha256(argv_bytes).hexdigest() != capacity_match.group("argv_sha256"):
        return None, "capacity-start skip server.argv SHA-256 mismatch"
    try:
        command = shlex.split(argv_bytes.decode("utf-8"), posix=True)
    except ValueError as exc:
        return None, f"capacity-start skip has unparsable server.argv: {exc}"
    prefix = (
        "env", "GGML_IQK=1", "LD_LIBRARY_PATH=/mnt/raid0/llm/llama.cpp/build-hip/bin",
        "taskset", "-c", "184-191", V8_BINARY,
    )
    if tuple(command[:len(prefix)]) != prefix:
        return None, "capacity-start skip server.argv does not use canonical v8 launch prefix"
    server_args = command[len(prefix):]
    expected_options = {
        "--host": "127.0.0.1",
        "--port": "18072",
        "-np": str(np),
        "-c": str(np * length),
        "-t": "8",
        "-tb": "8",
        "-b": "2048",
        "-ub": "2048",
        "-ctk": "f16",
        "-ctv": "f16",
        "--device": "ROCm0",
        "-ngl": "all",
        "-fa": "on",
        "--reasoning": "off",
    }
    binding = CANONICAL_SURFACE_BINDINGS.get(label)
    if binding is None:
        return None, "capacity-start skip label is not a canonical surface"
    expected_options["-m"] = binding[2]
    for option, expected in expected_options.items():
        positions = [index for index, value in enumerate(server_args) if value == option]
        if len(positions) != 1 or positions[0] + 1 >= len(server_args):
            return None, f"capacity-start skip server.argv has missing or duplicate {option}"
        if server_args[positions[0] + 1] != expected:
            if option == "-m":
                return None, "capacity-start skip server.argv -m is not canonical model"
            return None, f"capacity-start skip server.argv {option} is not {expected!r}"
    for flag in ("--metrics", "--slots", "--jinja"):
        if server_args.count(flag) != 1:
            return None, f"capacity-start skip server.argv has missing or duplicate {flag}"
    expected_mtp_depth = CANONICAL_MTP_DEPTHS[label]
    if mtp_depth != expected_mtp_depth:
        return None, "capacity-start skip provenance mtp_depth is not canonical"
    mtp_flags = ("--spec-type", "--spec-draft-n-max")
    if mtp_depth == 0:
        if any(flag in server_args for flag in mtp_flags):
            return None, "capacity-start skip server.argv has MTP flags for mtp_depth 0"
    else:
        for flag, expected in (("--spec-type", "draft-mtp"), ("--spec-draft-n-max", str(mtp_depth))):
            positions = [index for index, value in enumerate(server_args) if value == flag]
            if len(positions) != 1 or positions[0] + 1 >= len(server_args):
                return None, f"capacity-start skip server.argv has missing or duplicate {flag}"
            if server_args[positions[0] + 1] != expected:
                return None, f"capacity-start skip server.argv {flag} is not {expected!r}"
    return text.strip(), None


def publish_final(destination: Path, rendered: str) -> int:
    """Atomically create *destination*, or accept an identical prior publication.

    ``link`` provides no-replace visibility after a fully fsynced temporary file
    has been written on the same filesystem.  A competing publisher can only
    win with a complete file; its bytes must be identical for this invocation to
    be idempotent.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w") as output:
            output.write(rendered)
            output.flush()
            os.fsync(output.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError:
            if destination.read_text() != rendered:
                return 3
            return 0
        directory_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        return 0
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def inspect_surface(base: Path) -> dict[str, Any]:
    try:
        spec = load_spec(base)
    except ValueError as exc:
        return {"label": base.name, "state": "invalid", "reason": str(exc), "cells": []}

    cells: list[dict[str, Any]] = []
    terminal = True
    for np, length in spec.cells:
        cell_dir = base / f"np{np}_L{length}"
        result, skip = cell_dir / "results.json", cell_dir / "skip.txt"
        row: dict[str, Any] = {"np": np, "length": length}
        if result.exists() and skip.exists():
            row.update(state="invalid", reason="both results.json and skip.txt exist")
            terminal = False
        elif result.exists():
            throughput, reason = validate_result(result, spec.label, np, length)
            if reason:
                row.update(state="invalid", reason=reason)
                terminal = False
            else:
                row.update(state="measured", throughput=throughput)
        elif skip.is_file():
            reason, error = validate_skip(skip, spec.label, spec.mtp_depth, np, length)
            if error:
                row.update(state="invalid", reason=error)
                terminal = False
            else:
                row.update(state="skipped", reason=reason)
        else:
            row.update(state="missing")
            terminal = False
        cells.append(row)

    marker = base / "complete.txt"
    complete = marker.is_file() and COMPLETE_RE.fullmatch(marker.read_text()) is not None
    state = "terminal" if terminal and complete else "incomplete"
    return {
        "label": spec.label,
        "mode": spec.mode,
        "grid": spec.grid,
        "complete_marker": complete,
        "state": state,
        "cells": cells,
        "measured_cells": sum(row["state"] == "measured" for row in cells),
        "skipped_cells": sum(row["state"] == "skipped" for row in cells),
        "missing_cells": sum(row["state"] == "missing" for row in cells),
        "invalid_cells": sum(row["state"] == "invalid" for row in cells),
    }


def validate_cell(base: Path, np: int, length: int) -> str | None:
    """Validate exactly one canonical cell, including its disposition."""
    try:
        spec = load_spec(base)
    except ValueError as exc:
        return str(exc)
    if (np, length) not in spec.cells:
        return f"({np}, {length}) is not a canonical {spec.grid} cell"
    directory = base / f"np{np}_L{length}"
    result, skip = directory / "results.json", directory / "skip.txt"
    if result.exists() and skip.exists():
        return "both results.json and skip.txt exist"
    if result.exists():
        _, reason = validate_result(result, spec.label, np, length)
        return reason
    if skip.exists():
        _, reason = validate_skip(skip, spec.label, spec.mtp_depth, np, length)
        return reason
    return "cell has no terminal disposition"


def require_terminal_surface(base: Path, expected_grid: str) -> str | None:
    """Return an explanation unless *base* is a complete, exact grid."""
    report = inspect_surface(base)
    if report.get("grid") != expected_grid:
        return f"grid is {report.get('grid')!r}, expected {expected_grid!r}"
    if report.get("state") != "terminal":
        return (
            f"surface is {report.get('state')}; missing={report.get('missing_cells')} "
            f"invalid={report.get('invalid_cells')}"
        )
    return None


def make_report(root: Path, labels: list[str]) -> dict[str, Any]:
    surfaces = [inspect_surface(root / label) for label in labels]
    terminal = all(surface["state"] == "terminal" for surface in surfaces)
    return {
        "instrument": "np_context_study_v8_20260727",
        "root": str(root),
        "final_publishable": terminal,
        "surfaces": surfaces,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--label", action="append", dest="labels", help="surface label; repeatable")
    parser.add_argument("--write", type=Path, help="write final aggregate only when all selected surfaces are terminal")
    parser.add_argument("--require-terminal", action="store_true", help="exit nonzero unless every selected surface is exact and terminal")
    parser.add_argument("--validate-quality", type=Path, metavar="DIR", help="validate one saved rb1024 capture and exit")
    parser.add_argument("--quality-label")
    parser.add_argument("--quality-model")
    parser.add_argument("--quality-suite")
    parser.add_argument("--quality-n", type=int)
    parser.add_argument("--quality-max-tokens", type=int)
    parser.add_argument("--quality-questions")
    parser.add_argument("--quality-thinking", choices=("true", "false"))
    parser.add_argument("--validate-cell", nargs=3, metavar=("LABEL", "NP", "LENGTH"),
                        help="validate one canonical terminal cell and exit")
    parser.add_argument("--require-cells", action="store_true",
                        help="exit nonzero unless every selected canonical cell is exact; marker not required")
    args = parser.parse_args()
    if args.validate_quality:
        values = (
            args.quality_label, args.quality_model, args.quality_suite,
            args.quality_n, args.quality_max_tokens, args.quality_questions,
            args.quality_thinking,
        )
        if any(value is None for value in values):
            parser.error("--validate-quality requires every --quality-* argument")
        reason = validate_quality(
            args.validate_quality,
            label=args.quality_label,
            model=args.quality_model,
            suite_name=args.quality_suite,
            expected_n=args.quality_n,
            max_tokens=args.quality_max_tokens,
            questions=args.quality_questions,
            thinking=args.quality_thinking == "true",
        )
        if reason:
            print(reason, file=sys.stderr)
            return 2
        print("QUALITY_CAPTURE_VALID")
        return 0
    if args.validate_cell:
        label, np_text, length_text = args.validate_cell
        reason = validate_cell(args.root / label, int(np_text), int(length_text))
        if reason:
            print(reason, file=sys.stderr)
            return 2
        print("CELL_VALID")
        return 0
    labels = args.labels or list(CANONICAL_LABELS)
    if not labels:
        parser.error("no surface directories selected")
    report = make_report(args.root, labels)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    print(rendered, end="")
    if args.require_cells and any(
        surface["missing_cells"] or surface["invalid_cells"]
        for surface in report["surfaces"]
    ):
        return 2
    if args.require_terminal and not report["final_publishable"]:
        return 2
    if args.write:
        if not report["final_publishable"]:
            print("refusing --write: selected surfaces are not all terminal", file=sys.stderr)
            return 2
        result = publish_final(args.write, rendered)
        if result:
            print("refusing --write: destination exists with different content", file=sys.stderr)
            return result
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
