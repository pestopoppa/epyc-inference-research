#!/usr/bin/env python3
"""Fail-closed HIP matmul route capture and two-run A/A identity check.

The live entrypoint is intentionally dry by default.  The loop owner supplies the
stage claim and adds ``--execute`` only inside an approved correctness-surface
window; this module neither acquires hardware nor changes campaign state.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Callable, Mapping

from . import census, instruments, residency

SCHEMA = "epyc.autokernel.hip_route_aa.v1"
OBSERVED = census.OBSERVED
UNKNOWN = census.UNKNOWN
ROUTES = frozenset({"CUBLAS_PRECHECK", "MMVF", "MMF", "MMVQ", "MMQ", "CUBLAS",
                    "MMVF_MMID", "MMF_MMID", "MMVQ_MMID", "MMQ_MMID", "CUBLAS_MMID"})
MARKER = "GGML_CUDA_MUL_MAT_ROUTE"
ROUTE_LINE = re.compile(
    r"GGML_CUDA_MUL_MAT_ROUTE\s+route=(?P<route>[A-Z0-9_]+)\s+"
    r"src0=\S+\s+src1=\S+\s+dst=\S+\s+type=(?P<type>[a-zA-Z0-9_]+)\s+"
    r"cc=\d+\s+src0_ne=\[[^\]]+\]\s+src1_ne=\[[^\]]+\]\s+"
    r"dst_ne=\[[^\]]+\]\s+ne11=(?P<ne11>\d+)\s*$"
)
IDENTITY_FIELDS = frozenset({"workload_id", "workload_signature", "model", "recipe",
                             "kernel", "host"})


def _digest(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def parse_hip_matmul_routes(log: str) -> dict[str, Any]:
    """Return a stable ``(route,type,ne11)`` multiset or UNKNOWN on bad evidence.

    Unrelated llama logging is ignored.  Every line containing our marker must
    parse completely, so a changed/truncated producer format cannot silently
    disappear from the census.
    """
    plain = census.ANSI.sub("", log)
    counts: Counter[tuple[str, str, int]] = Counter()
    errors: list[dict[str, Any]] = []
    for number, line in enumerate(plain.splitlines(), 1):
        if MARKER not in line:
            continue
        match = ROUTE_LINE.search(line)
        if match is None or match.group("route") not in ROUTES:
            errors.append({"line": number, "reason": "malformed_or_unknown_route"})
            continue
        counts[(match.group("route"), match.group("type"), int(match.group("ne11")))] += 1
    rows = [{"route": route, "type": tensor_type, "ne11": ne11, "count": count}
            for (route, tensor_type, ne11), count in sorted(counts.items())]
    observed = bool(rows) and not errors
    return {"state": OBSERVED if observed else UNKNOWN,
            "hip_matmul_routes": rows,
            "route_events": sum(row["count"] for row in rows),
            "parse_errors": errors,
            "multiset_sha256": _digest(rows) if observed else None,
            "vacuous_guards": {"route_events_gt_zero": bool(rows),
                                "all_marker_lines_parsed": not errors}}


def aa_identity(first: Mapping[str, Any], second: Mapping[str, Any],
                identity: Mapping[str, Any]) -> dict[str, Any]:
    """Bind two captures to one caller-supplied build/recipe identity."""
    missing = IDENTITY_FIELDS - set(identity)
    if missing:
        raise ValueError(f"identity missing required fields: {', '.join(sorted(missing))}")
    same = (first.get("state") == OBSERVED and second.get("state") == OBSERVED
            and first.get("hip_matmul_routes") == second.get("hip_matmul_routes"))
    return {"schema": SCHEMA, "identity": dict(identity),
            "identity_sha256": _digest(identity), "runs": [dict(first), dict(second)],
            "state": OBSERVED if same else UNKNOWN,
            "structurally_identical": same}


def run_aa(binary: Path, model: Path, identity: Mapping[str, Any], *,
           shape: census.Shape = census.Shape("decode", 1),
           recipe_argv: tuple[str, ...] = (), timeout: int = 900,
           runner: Callable[..., subprocess.CompletedProcess] = census.run_process
           ) -> dict[str, Any]:
    """Run the exact same route probe twice, sequentially, and compare multisets."""
    env = dict(residency.loader_env(binary))
    env["GGML_CUDA_LOG_MMVQ_ROUTE"] = "2"
    argv = census.shape_argv(binary, model, shape, recipe_argv)
    captures = []
    for _ in range(2):
        proc = runner(argv, env=env, timeout=timeout)
        parsed = parse_hip_matmul_routes(proc.stdout + proc.stderr)
        if proc.returncode != 0:
            parsed["state"] = UNKNOWN
        parsed["rc"] = proc.returncode
        captures.append(parsed)
    return aa_identity(captures[0], captures[1], identity)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="autokernel.loop.hip_routes")
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--identity-json", required=True, type=Path,
                        help="build/recipe/kernel/DSO identity JSON; shared by both A/A arms")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--recipe-arg", action="append", default=[])
    instruments.add_posture_args(parser)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    posture = instruments.resolve_posture(args)
    if posture.dry_run:
        print("DRY RUN: two sequential identical decode route captures; pass --execute "
              "only while holding the owner claim")
        return 0
    try:
        identity = json.loads(args.identity_json.read_text())
        if not isinstance(identity, dict) or not identity:
            raise ValueError("identity JSON must be a non-empty object")
        instruments.require_binary(args.binary.parent.parent, args.binary.name)
        result = run_aa(args.binary, args.model, identity,
                        recipe_argv=tuple(args.recipe_arg), timeout=args.timeout)
    except (OSError, ValueError, json.JSONDecodeError, instruments.InstrumentRefusal) as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return instruments.REFUSED
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0 if result["state"] == OBSERVED else instruments.REFUSED


if __name__ == "__main__":
    raise SystemExit(main())
