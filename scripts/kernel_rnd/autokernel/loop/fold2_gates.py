#!/usr/bin/env python3
"""The FOLD-2 correctness/perf gate battery, run by hand with the loop dead and GPU free.

    python3 -m autokernel.loop.fold2_gates \
        --candidate-build /mnt/raid0/llm/tmp/build-fold-ef81196d5 \
        --anchor-build /mnt/raid0/llm/autokernel/loop-memory/anchor-gen-021 \
        [--only-correctness] [--execute]

Gates, in order (any FAIL stops the fold; nothing is fast-forwarded by this module):

  G1  `test-backend-ops -b ROCm0 -o SSM_SCAN` -- includes the upstream K=4 / K=3 rollback
      cases. A case the CUDA guard DECLINES (K>1 -> supports_op false) is reported by the
      harness as NOT SUPPORTED, which is the port's intended behaviour and NOT a failure;
      a case that runs and mismatches IS a failure.
  G2  `test-backend-ops -b ROCm0 -o MUL_MAT` -- the loop's standing oracle.
  G3  `test-backend-ops -b ROCm0 -o GATED_DELTA_NET` -- the 27B's recurrent op, if the
      harness names it.
  G4  dispatch OBSERVED: llama-bench on the 27B under `GGML_SCHED_DEBUG=2`. The graph must
      contain NO SSM_SCAN node (qwen35 is gated-delta-net) and its GATED_DELTA_NET /
      recurrent nodes must be assigned to ROCm0, never to the CPU backend.
  G5  tg128 paired A/B: candidate vs the champion build. PASS if not decisively NEGATIVE
      (the CPU levers must not cost the GPU headline); the measured effect is recorded
      either way.

THE VACUOUS-PASS GUARDS ARE THE POINT OF THIS MODULE, NOT DECORATION. Both of them were
written after the gate reported PASS having proved nothing:

  * G1-G3's tally is parsed from ANSI-STRIPPED text and requires `ok > 0` and the
    harness's own `N/M tests passed` line to agree. The first run reported PASS with OK=0
    because the harness COLOURS its verdicts, so `": OK"` never matched -- zero passes and
    zero failures read as "no failures".
  * G4 requires `-v` on llama-bench (which otherwise swallows the scheduler's log lines --
    the first run parsed an EMPTY log and "passed"), and requires that a real graph was
    actually observed: >1000 nodes parsed and at least one recurrent op present. A PASS
    with zero nodes or zero recurrent ops is a broken probe, not a clean graph.

Every parser and every verdict below is a pure function over text, so those guards are
tested against real coloured harness output and against output produced WITHOUT `-v`, and
the process calls are two thin seams a test can stub. Dry by default; `--execute` runs it.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys
import time

from . import bench, instruments, residency

#: The harness colours its verdicts. Strip before ANY counting -- see the module docstring.
ANSI = re.compile(r"\x1b\[[0-9;]*m")

#: `node #  0 (  GET_ROWS): name (size) [  CPU  ] ...` -- op name and assigned backend.
NODE_LINE = re.compile(r"^node #\s*\d+\s*\(\s*([A-Z_0-9]+)\s*\):.*?\[\s*([A-Za-z0-9]+)\s*\]",
                       flags=re.M)

#: Ops that carry the 27B's recurrent state. At least one must appear, and none of them
#: may be assigned to the CPU backend.
RECURRENT_OP = re.compile(r"GATED_DELT|SSM_CONV|SOLVE_TRI|RWKV")

#: A real 27B graph is thousands of nodes. This is the "did we observe anything at all"
#: bar that the missing `-v` defeated.
MIN_GRAPH_NODES = 1000

#: The only op allowed to land on the CPU backend in a fully offloaded graph.
ALLOWED_CPU_OPS = frozenset({"GET_ROWS"})

#: The device string that proves the probe reached the GPU rather than a CPU-only build.
DEVICE_MARKER = "MI210"


def run_process(argv: list[str], *, env: dict, timeout: int) -> subprocess.CompletedProcess:
    """THE PROCESS SEAM for G1-G4. Stubbed by every test in this package."""
    return subprocess.run(argv, capture_output=True, text=True, timeout=timeout, env=env)


def compare_tg128(anchor_binary: Path, candidate_binary: Path, model: Path, *,
                  pairs: int, floor_pct: float) -> dict:
    """THE MEASUREMENT SEAM for G5 -- the loop's own paired A/B, tg128 surface."""
    pp, tg, ubatch = bench.SURFACES["tg128"]
    return bench.compare(bench.Arm("champion", anchor_binary),
                         bench.Arm("fold_candidate", candidate_binary),
                         model, pp=pp, tg=tg, pairs=pairs, noise_floor_pct=floor_pct,
                         surface="tg128", ubatch=ubatch, calibrated=True).to_dict()


def tally(text: str) -> dict:
    """Count the harness's verdicts out of ANSI-STRIPPED output.

    Stripping first is load-bearing, not tidiness: with the escape codes in place `": OK"`
    never matches, so a fully passing run counts as ok=0 fail=0 -- which any "no failures"
    rule reads as a pass. `passed`/`total` come from the harness's OWN tally line so the
    counting here can be cross-checked against the counting there.
    """
    plain = ANSI.sub("", text)
    match = re.search(r"(\d+)/(\d+) tests passed", plain)
    passed, total = (int(match.group(1)), int(match.group(2))) if match else (0, 0)
    return {"ok": len(re.findall(r": OK\b", plain)),
            "fail": len(re.findall(r": FAIL\b", plain)),
            "not_supported": len(re.findall(r"not supported", plain)),
            "passed": passed, "total": total,
            "tail": "\n".join(plain.strip().splitlines()[-4:])}


def backend_op_verdict(returncode: int, counts: dict) -> str:
    """VACUOUS-PASS GUARD (2026-09-08). A gate PASSES only if at least one case ACTUALLY
    RAN and passed, and the harness's own tally agrees with ours. `rc == 0 and fail == 0`
    alone is what a run that counted nothing looks like."""
    ran_something = counts["ok"] > 0 and counts["total"] > 0
    agrees = counts["passed"] == counts["total"]
    return ("PASS" if returncode == 0 and counts["fail"] == 0 and ran_something and agrees
            else "FAIL")


def backend_op_argv(binary: Path, op: str) -> list[str]:
    return [str(binary), "test", "-o", op, "-b", "ROCm0", "-j", "1"]


def dispatch_argv(binary: Path, model: Path) -> list[str]:
    """llama-bench under the scheduler debug log.

    `-v` is REQUIRED: llama-bench swallows the scheduler's log lines otherwise, and the
    first run of this gate parsed an empty log and "passed". It is in the argv builder,
    not at the call site, so a test can assert it is still there.
    """
    return ["taskset", "-c", bench.CPU_LIST, str(binary), "-m", str(model), "-p", "0",
            "-n", "16", "-ngl", "99", "-r", "1", "-o", "json", "-v"]


def parse_scheduler_graph(log: str) -> dict:
    """What the scheduler actually assigned, as counts per (op, backend).

    Pure and fail-closed by construction: an empty log yields zero nodes, which the
    verdict below refuses. It does not "assume the graph was fine because nothing looked
    wrong" -- nothing looking wrong is exactly what an unobserved graph looks like.
    """
    ops: dict[str, dict[str, int]] = {}
    nodes = NODE_LINE.findall(log)
    for op, backend in nodes:
        ops.setdefault(op, {}).setdefault(backend, 0)
        ops[op][backend] += 1
    recurrent = {op: backends for op, backends in ops.items() if RECURRENT_OP.match(op)}
    return {"nodes_parsed": len(nodes),
            "ssm_scan_nodes": sum(ops.get("SSM_SCAN", {}).values()),
            "recurrent_ops": recurrent,
            "recurrent_nodes": sum(sum(b.values()) for b in recurrent.values()),
            "recurrent_on_cpu": sum(b.get("CPU", 0) for b in recurrent.values()),
            "cpu_assigned_ops": {op: b["CPU"] for op, b in ops.items() if b.get("CPU")},
            "device_seen": DEVICE_MARKER in log}


def dispatch_verdict(returncode: int, graph: dict) -> str:
    """VACUOUS-PASS GUARD: the graph must have been OBSERVED before it can be clean.

    `nodes_parsed > MIN_GRAPH_NODES` and `recurrent_nodes > 0` are the observation
    requirement (a probe run without `-v` produces neither); the rest is the actual
    dispatch rule -- no SSM_SCAN at all, no recurrent op on the CPU backend, and nothing
    but GET_ROWS left on the CPU.
    """
    return ("PASS" if (returncode == 0 and graph["device_seen"]
                       and graph["nodes_parsed"] > MIN_GRAPH_NODES
                       and graph["recurrent_nodes"] > 0
                       and graph["ssm_scan_nodes"] == 0
                       and graph["recurrent_on_cpu"] == 0
                       and set(graph["cpu_assigned_ops"]) <= ALLOWED_CPU_OPS)
            else "FAIL")


def ab_verdict(comparison: dict) -> str:
    """G5: FAIL only on a DECISIVE negative. A within-floor move is not evidence the CPU
    levers cost the GPU headline, and refusing on it would refuse noise."""
    return ("FAIL" if bool(comparison["decisive"]) and comparison["effect_pct"] < 0
            else "PASS")


def run_backend_op(candidate_build: Path, op: str, *, timeout: int = 1800,
                   echo=print) -> dict:
    binary = candidate_build / "bin" / "test-backend-ops"
    started = time.time()
    proc = run_process(backend_op_argv(binary, op), env=residency.loader_env(binary),
                       timeout=timeout)
    counts = tally(proc.stdout + proc.stderr)
    verdict = backend_op_verdict(proc.returncode, counts)
    echo(f"  {op:16s} rc={proc.returncode} OK={counts['ok']} FAIL={counts['fail']} "
         f"not-supported={counts['not_supported']} [{time.time() - started:.0f}s] "
         f"-> {verdict}\n    {counts['tail'][:300]}")
    return {"rc": proc.returncode, "ok": counts["ok"], "fail": counts["fail"],
            "not_supported": counts["not_supported"], "passed": counts["passed"],
            "total": counts["total"], "verdict": verdict, "tail": counts["tail"]}


def run_dispatch_probe(candidate_build: Path, model: Path, *, timeout: int = 900,
                       echo=print) -> dict:
    binary = candidate_build / "bin" / "llama-bench"
    env = dict(residency.loader_env(binary))
    env["GGML_SCHED_DEBUG"] = "2"
    proc = run_process(dispatch_argv(binary, model), env=env, timeout=timeout)
    graph = parse_scheduler_graph(proc.stdout + proc.stderr)
    row = {**graph, "rc": proc.returncode}
    row["verdict"] = dispatch_verdict(proc.returncode, graph)
    echo(f"  rc={proc.returncode} device={graph['device_seen']} "
         f"nodes={graph['nodes_parsed']} SSM_SCAN={graph['ssm_scan_nodes']} "
         f"recurrent={graph['recurrent_ops']} on-CPU={graph['recurrent_on_cpu']} "
         f"cpu-ops={graph['cpu_assigned_ops']} -> {row['verdict']}")
    return row


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="autokernel.loop.fold2_gates", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--candidate-build", type=Path, required=True)
    parser.add_argument("--anchor-build", type=Path, default=None,
                        help="the champion build G5 compares against. Required unless "
                             "--only-correctness.")
    parser.add_argument("--candidate-id", default=None,
                        help="label recorded in the result (default: the candidate "
                             "build directory's name)")
    parser.add_argument("--model", type=Path, default=instruments.DEFAULT_MODEL)
    parser.add_argument("--out", type=Path,
                        default=instruments.DEFAULT_STORE / "fold2-result.json",
                        help="result JSON (default: %(default)s)")
    parser.add_argument("--pairs", type=int, default=20, help="G5 alternating pairs")
    parser.add_argument("--floor", type=float, default=0.638,
                        help="G5 tg128 noise floor %%, calibrated 2026-09-04 at 20 pairs")
    parser.add_argument("--only-correctness", action="store_true",
                        help="G1-G4 only; G5 (comparative) after the keep decision")
    instruments.add_posture_args(parser)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    posture = instruments.resolve_posture(args)
    candidate_id = args.candidate_id or Path(args.candidate_build).name
    try:
        instruments.require_binary(args.candidate_build, "test-backend-ops")
        instruments.require_binary(args.candidate_build, "llama-bench")
        if not args.only_correctness:
            if args.anchor_build is None:
                raise instruments.InstrumentRefusal(
                    "G5 compares against the champion build: pass --anchor-build, or "
                    "--only-correctness to run G1-G4 alone")
            instruments.require_binary(args.anchor_build, "llama-bench")
    except instruments.InstrumentRefusal as refusal:
        print(f"REFUSED: {refusal}", file=sys.stderr)
        return instruments.REFUSED
    print(f"candidate {candidate_id} ({args.candidate_build})")
    print(f"model     {args.model}")
    g5_note = ("" if args.only_correctness
               else f" + G5 ({args.pairs} pairs, floor {args.floor}%)")
    print(f"gates     G1-G4{g5_note}")
    print(f"posture   {posture.describe()}")
    if posture.dry_run:
        print("\nDRY RUN -- nothing launched. The gates that would run:")
        binary = Path(args.candidate_build) / "bin" / "test-backend-ops"
        for op in ("SSM_SCAN", "MUL_MAT", "GATED_DELTA_NET"):
            print("  " + " ".join(backend_op_argv(binary, op)))
        print("  GGML_SCHED_DEBUG=2 " + " ".join(
            dispatch_argv(Path(args.candidate_build) / "bin" / "llama-bench", args.model)))
        print("Pass --execute to run them.")
        return 0

    result = {"candidate": candidate_id, "gates": {}}
    print("=== G1 SSM_SCAN (port correctness incl. K>1 rollback cases)")
    result["gates"]["G1_ssm_scan"] = run_backend_op(args.candidate_build, "SSM_SCAN")
    print("=== G2 MUL_MAT (standing oracle)")
    result["gates"]["G2_mul_mat"] = run_backend_op(args.candidate_build, "MUL_MAT",
                                                   timeout=3600)
    print("=== G3 GATED_DELTA_NET (27B recurrent op)")
    result["gates"]["G3_gdn"] = run_backend_op(args.candidate_build, "GATED_DELTA_NET")
    print("=== G4 dispatch observed on the 27B graph")
    result["gates"]["G4_dispatch"] = run_dispatch_probe(args.candidate_build, args.model)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.only_correctness:
        result["overall_correctness"] = (
            "PASS" if all(g["verdict"] == "PASS" for g in result["gates"].values())
            else "FAIL")
        args.out.write_text(json.dumps(result, indent=2))
        print(f"FOLD-2 CORRECTNESS (G1-G4): {result['overall_correctness']} -> {args.out}")
        return 0 if result["overall_correctness"] == "PASS" else 1

    print(f"=== G5 tg128 A/B candidate vs champion, {args.pairs} pairs, "
          f"floor {args.floor}%")
    comparison = compare_tg128(Path(args.anchor_build) / "bin" / "llama-bench",
                               Path(args.candidate_build) / "bin" / "llama-bench",
                               args.model, pairs=args.pairs, floor_pct=args.floor)
    g5 = {"effect_pct": comparison["effect_pct"], "decisive": comparison["decisive"],
          "drifting": comparison.get("drifting"), "verdict": ab_verdict(comparison)}
    print(f"  effect {comparison['effect_pct']:+.3f}% decisive={comparison['decisive']} "
          f"drifting={comparison.get('drifting')} -> {g5['verdict']}")
    result["gates"]["G5_tg128_ab"] = g5
    result["g5_full"] = comparison
    result["overall"] = ("PASS" if all(g["verdict"] == "PASS"
                                       for g in result["gates"].values()) else "FAIL")
    args.out.write_text(json.dumps(result, indent=2))
    print(f"FOLD-2 OVERALL: {result['overall']}  -> {args.out}")
    return 0 if result["overall"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
