"""AKX-ALL-31.NATIVE: host side of the native `AK_NUMERICS_V1` producer.

The producer is the C++ `test-akx-numerics` (llama.cpp tests/, experimental branch). This module
(1) writes the planted inputs it consumes (`write_inputs`, from `numerics_gates.non_finite_inputs()` and
`accum_inputs()` so the native and host legs judge IDENTICAL data) and (2) judges its stdout
(`judge`): the incumbent receipt must pass both gates, and each native planted control must go red on
ITS gate and green on the other. Absence of a receipt, a missing variant, or a control that passes is a
failure, never a pass. Reuses numerics_gates' receipt convention (parse_receipts / verdict_from_receipt).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from typing import Any, Sequence

from . import numerics_gates as ng

VARIANTS = ("incumbent", "control_clamp_max", "control_f16_dot")
#: control variant -> (gate that MUST be red, gate that must stay green)
CONTROL_RED = {"control_clamp_max": (ng.GATE_NON_FINITE, ng.GATE_FP32_ACCUM),
               "control_f16_dot": (ng.GATE_FP32_ACCUM, ng.GATE_NON_FINITE)}


def _tok(v: float) -> str:
    if math.isnan(v):
        return "nan"
    if math.isinf(v):
        return "inf" if v > 0 else "-inf"
    return repr(float(v))


def inputs_text() -> str:
    rows = ng.non_finite_inputs()
    A, B = ng.accum_inputs()
    out = [f"MAX {len(rows)} {len(rows[0])}"]
    out += [" ".join(_tok(v) for v in r) for r in rows]
    out.append(f"DOT {len(A)} {len(A[0])}")
    out += [" ".join(_tok(v) for v in r) for r in A]
    out += [" ".join(_tok(v) for v in r) for r in B]
    return "\n".join(out) + "\n"


def write_inputs(path: str) -> None:
    with open(path, "w") as f:
        f.write(inputs_text())


def judge(text: str) -> dict[str, Any]:
    """Judge the native stdout. Returns {ok, backend, variants: {name: [verdict dicts]}, problems}."""
    by_variant: dict[str, dict] = {}
    for r in ng.parse_receipts(text):
        if isinstance(r, dict) and r.get("variant") in VARIANTS:
            by_variant.setdefault(r["variant"], r)
    problems: list[str] = []
    verdicts: dict[str, list[dict]] = {}
    for name in VARIANTS:
        r = by_variant.get(name)
        if r is None:
            problems.append(f"receipt_missing: {name}")
            continue
        vs = ng.verdict_from_receipt(r)
        verdicts[name] = [{"gate": v.gate, "passed": v.passed, "reason": v.reason} for v in vs]
        passed = {v.gate: v.passed for v in vs}
        if name == "incumbent":
            for v in vs:
                if not v.passed:
                    problems.append(f"incumbent_failed_{v.gate}: {v.reason}")
        else:
            red, green = CONTROL_RED[name]
            if passed.get(red, True):
                problems.append(f"control_not_red: {name} passed {red}")
            if not passed.get(green, False):
                problems.append(f"control_collateral_red: {name} failed {green}")
    backends = sorted({str(r.get("backend")) for r in by_variant.values()})
    return {"ok": not problems, "backend": backends, "variants": verdicts, "problems": problems}


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("write-inputs"); w.add_argument("path")
    j = sub.add_parser("judge"); j.add_argument("log"); j.add_argument("--out")
    a = ap.parse_args(argv)
    if a.cmd == "write-inputs":
        write_inputs(a.path)
        return 0
    res = judge(open(a.log, errors="replace").read())
    if a.out:
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1)
    print(json.dumps(res))
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
