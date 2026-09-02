#!/usr/bin/env python3
"""Helper for boundary_20260901.sh — the run-23→24 overnight boundary driver.

Four subcommands, all invoked by the driver (never run these by hand mid-boundary):

  probe    Dry-run precondition lines: rocprofv3 resolvable, claim module importable,
           GPU lock path. Prints [ SAT ]/[UNSAT] rows; always exits 0 (reporting,
           not gating).

  rocprof  One rocprofv3 --kernel-trace llama-bench invocation for a confirm
           surface (dec-b4 / dec-b8) on the FROZEN production build, under the
           mi210_0 claim (claim.hold — acquired for the window, never observed).
           Writes the ranked per-kernel dispatch table JSON: the rung identity
           artifact. Reuses the loop's own hotspots module for the vendored-SDK
           environment and the trace parser; the argv is re-built here because
           hotspots.profile() takes no ubatch and the dec-b* surfaces are defined
           by -b N -ub N (bench.SURFACES).

  smoke    DFlash2 drafter-head llama-bench smoke: one tg128-style and one
           dec-b4-style invocation (champion_anchor_validation.py bench pattern:
           taskset 184-191, numactl --interleave=all, own-bin-first loader path).
           Exit 0 = clean exits + sane tok/s on both; anything else exits 1 and
           the driver records-and-continues (informs D5, does not gate).

  report   Write the run-24 readiness package. Includes the exact two-rung run-24
           command and the all-green verdict; the LAUNCH decision stays in the
           driver (PREAUTH_RUN24 token + all green), never here.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

HERE = Path(__file__).resolve().parent

SURFACE_SHAPES = {  # bench.SURFACES, pinned here for the frozen-build invocations
    "dec-b4": (512, 0, 4),
    "dec-b8": (512, 0, 8),
}


def _lane_modules(lane_root: Path):
    """Import claim + hotspots from the lane tree (post-merge code at run time)."""
    sys.path.insert(0, str(lane_root / "scripts" / "kernel_rnd"))
    from autokernel.loop import claim, hotspots  # noqa: PLC0415
    return claim, hotspots


# ----------------------------------------------------------------------- probe
def cmd_probe(args: argparse.Namespace) -> int:
    try:
        claim, hotspots = _lane_modules(Path(args.lane_root))
    except Exception as exc:  # noqa: BLE001
        print(f"[UNSAT] lane claim/hotspots import        {exc}")
        return 0
    rocprof = hotspots._resolve_rocprof()  # noqa: SLF001 — the loop's own resolver
    if rocprof:
        print(f"[ SAT ] rocprofv3                          {rocprof}")
    else:
        print("[UNSAT] rocprofv3                          none of: "
              + ", ".join(hotspots.ROCPROF_CANDIDATES))
    lock = claim.DEVICE_LOCK
    print(f"[ SAT ] mi210_0 claim path                 {lock} "
          f"({'exists' if lock.exists() else 'will be created on first hold'})")
    return 0


# --------------------------------------------------------------------- rocprof
def cmd_rocprof(args: argparse.Namespace) -> int:
    claim, hotspots = _lane_modules(Path(args.lane_root))
    binary, model, out = Path(args.binary), Path(args.model), Path(args.out)
    pp, tg, ubatch = SURFACE_SHAPES[args.surface]

    rocprof = hotspots._resolve_rocprof()  # noqa: SLF001
    if rocprof is None:
        print("REFUSED: no rocprofv3 found; tried "
              + ", ".join(hotspots.ROCPROF_CANDIDATES), file=sys.stderr)
        return 1

    with tempfile.TemporaryDirectory() as tmp:
        trace_dir = Path(tmp)
        argv = [rocprof, "--kernel-trace", "-d", str(trace_dir), "-o", "trace", "--",
                str(binary), "-m", str(model), "-p", str(pp), "-n", str(tg),
                "-b", str(ubatch), "-ub", str(ubatch),
                "-r", "1", "-ngl", "99", "-fa", "1", "-o", "json"]
        env = hotspots._profiler_env(binary, rocprof)  # noqa: SLF001
        print(f"[{time.strftime('%H:%M:%S')}] {args.surface} dispatch capture\n"
              f"  $ {' '.join(argv)}", flush=True)
        started = time.monotonic()
        with claim.hold() as receipt:  # acquired for the whole window, never observed
            print(f"  claim held on {receipt['device_id']}")
            done = subprocess.run(argv, capture_output=True, text=True,
                                  timeout=args.timeout_s, env=env)
        if done.returncode != 0:
            print(f"REFUSED: rocprofv3 rc={done.returncode}: "
                  f"{(done.stderr or done.stdout)[-400:]}", file=sys.stderr)
            return 1
        traces = (sorted(trace_dir.rglob("*kernel_trace.csv"))
                  or sorted(trace_dir.rglob("*.csv")))
        if not traces:
            print("REFUSED: rocprofv3 produced no kernel trace csv", file=sys.stderr)
            return 1
        rows = hotspots.parse_kernel_trace(
            traces[0].read_text(encoding="utf-8"), limit=args.limit)
    if not rows:
        print("REFUSED: kernel trace parsed to an empty dispatch table",
              file=sys.stderr)
        return 1

    body = {
        "schema": "epyc.autokernel.rung_identity_dispatch.v1",
        "surface": args.surface,
        "shape": {"pp": pp, "tg": tg, "ubatch": ubatch},
        "binary": str(binary),
        "model": str(model),
        "rocprof": rocprof,
        "elapsed_s": round(time.monotonic() - started, 1),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dispatch_table": [r.to_dict() for r in rows],
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")
    print(f"  {len(rows)} kernels; top: {rows[0].signature[:70]}")
    print(f"  wrote {out}")
    return 0


# ----------------------------------------------------------------------- smoke
def _bench_once(binary: Path, model: Path, *, pp: int, tg: int,
                ubatch: int | None, timeout_s: int) -> float:
    """champion_anchor_validation.py bench pattern, verbatim shape."""
    argv = ["taskset", "-c", "184-191", "numactl", "--interleave=all",
            str(binary), "-m", str(model),
            "-p", str(pp), "-n", str(tg), "-r", "1", "-ngl", "99", "-fa", "1",
            "-o", "json"]
    if ubatch:
        argv += ["-b", str(ubatch), "-ub", str(ubatch)]
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = f"{binary.parent}:/opt/rocm/lib"
    env.pop("HSA_OVERRIDE_GFX_VERSION", None)
    done = subprocess.run(argv, capture_output=True, text=True,
                          timeout=timeout_s, env=env)
    if done.returncode != 0:
        raise RuntimeError(f"llama-bench rc={done.returncode}: {done.stderr[-300:]}")
    rows = json.loads(done.stdout)
    key, want = ("n_prompt", pp) if pp else ("n_gen", tg)
    for row in rows:
        if int(row[key]) == want:
            return float(row["avg_ts"])
    raise RuntimeError(f"no row for {key}={want}")


def cmd_smoke(args: argparse.Namespace) -> int:
    claim, _ = _lane_modules(Path(args.lane_root))
    binary, model, out = Path(args.binary), Path(args.model), Path(args.out)
    results: dict[str, dict] = {}
    ok = True
    with claim.hold() as receipt:
        print(f"claim held on {receipt['device_id']}")
        for label, pp, tg, ubatch in (("tg128", 0, 128, None),
                                      ("dec-b4", 512, 0, 4)):
            try:
                tok_s = _bench_once(binary, model, pp=pp, tg=tg, ubatch=ubatch,
                                    timeout_s=args.timeout_s)
                sane = tok_s > 1.0
                results[label] = {"tok_s": tok_s, "sane": sane}
                print(f"  {label}: {tok_s:.2f} tok/s ({'sane' if sane else 'NOT SANE'})")
                ok = ok and sane
            except Exception as exc:  # noqa: BLE001 — smoke records, never raises out
                results[label] = {"error": str(exc)}
                print(f"  {label}: FAILED {exc}")
                ok = False
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "schema": "epyc.autokernel.dflash2_smoke.v1",
        "binary": str(binary), "model": str(model),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "passed": ok, "invocations": results,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    return 0 if ok else 1


# ---------------------------------------------------------------------- report
def _state(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            if "=" in line:
                key, _, value = line.partition("=")
                out[key] = value  # last write wins
    return out


def _state_all(path: Path, key: str) -> list[str]:
    if not path.is_file():
        return []
    return [line.partition("=")[2]
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.startswith(f"{key}=")]


def cmd_report(args: argparse.Namespace) -> int:
    state = _state(Path(args.state))
    work = Path(args.work_dir)
    store = Path(args.store)
    stem = Path(args.model27).stem
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    def done(key: str) -> str:
        return "DONE" if state.get(key) == "done" else "NOT DONE"

    steps = [
        ("wait for T0", done("step0"), "-", f"{work}/driver.log"),
        ("stop run 23", done("step1"),
         f"{state.get('step1_outcome', '-')}, escalation "
         f"{state.get('step1_escalated', 'n/a')}",
         state.get("step1_tally", f"{work}/step1-stop.log")),
        ("merge gate + suite", done("step2"),
         f"lane {state.get('lane_tip_pre_merge', '?')[:12]} -> "
         f"{state.get('lane_tip_post_merge', '?')[:12]}, "
         f"{state.get('step2_rung_commits', '?')} rung commit(s)",
         f"{work}/step2-merge.log"),
        ("re-anchored seeds", done("step3"),
         f"{state.get('step3_copied', '?')} file(s)",
         f"{work}/step3-seeds-manifest.txt"),
        ("rocprof rung identity", done("step4"), "-",
         f"{work}/rung-identity/"),
        ("DFlash2 smoke (non-gating)", done("step5"),
         state.get("step5_outcome", "-"), f"{work}/step5-dflash-smoke.json"),
        ("27B A/A calibration", done("step6"),
         f"anchor {state.get('step6_anchor', '?')}",
         f"{store}/calibration/<surface>.{stem}.json"),
    ]
    table = "\n".join(f"| {label} | {status} | {extra} | `{evidence}` |"
                      for label, status, extra, evidence in steps)

    floors = []
    for surface in ("dec-b4", "dec-b8"):
        path = store / "calibration" / f"{surface}.{stem}.json"
        if path.is_file():
            try:
                floor = json.loads(path.read_text(encoding="utf-8"))["floor_pct"]
                floors.append(f"- {surface}: " + ", ".join(
                    f"{k}p: {float(v):.3f}%"
                    for k, v in sorted(floor.items(), key=lambda kv: int(kv[0])))
                    + f"  (`{path}`)")
            except Exception as exc:  # noqa: BLE001
                floors.append(f"- {surface}: UNREADABLE ({path}: {exc})")
        else:
            floors.append(f"- {surface}: NOT CALIBRATED ({path} missing)")

    allgreen = state.get("allgreen", "unknown")
    reasons = _state_all(Path(args.state), "allgreen_reason")
    if allgreen == "yes":
        allgreen_block = (
            "**ALL GREEN.** Every gating step met its gate (step 5 is informative "
            "by design). With the PREAUTH_RUN24 token present the driver launches "
            "run 24 itself; without it, the command below is verified-ready.")
    elif allgreen == "no":
        allgreen_block = ("**NOT all green — HOLD.** Run 24 must NOT start. "
                          "Red steps:\n\n" + "\n".join(f"- {r}" for r in reasons))
    else:
        allgreen_block = ("**UNEVALUATED — HOLD.** The driver never reached the "
                          "all-green check.")

    launch = state.get("run24_launch", "not attempted")
    run24_cmd = Path(args.run24_cmd_file).read_text(encoding="utf-8") \
        if Path(args.run24_cmd_file).is_file() else "(command file missing)"

    report = f"""# Run-23 → Run-24 boundary readiness — 2026-09-01 overnight

Generated {now} by `scripts/benchmark/boundary_20260901.sh` (unattended).
Driver log: `{work}/driver.log` · state: `{args.state}`

## Per-step verdicts

| Step | State | Outcome | Evidence |
|---|---|---|---|
{table}

Run 23: pid {state.get('run23_pid', '?')} — {state.get('step1_outcome', '?')}; \
final tally: `{state.get('step1_tally', '?')}`.

## Keyed 27B calibration floors (as written to the store)

{chr(10).join(floors)}

## All-green conjunction verdict

{allgreen_block}

## Run-24 launch state

**{launch}** — the driver launches run 24 ONLY when `{work}/PREAUTH_RUN24`
exists AND every gating step is green (final refusal check: the loop's own
`--dry-run` must print `— verified`). Otherwise it stops at this package and
the run start stays operator-gated.

## Proposed run-24 command (two-rung: screen 1.5B, confirm 27B, pairs 5, dec-b4+dec-b8)

```bash
{run24_cmd}```
"""
    Path(args.readiness).write_text(report, encoding="utf-8")
    print(f"wrote {args.readiness}")
    return 0


# ------------------------------------------------------------------------ main
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("probe")
    p.add_argument("--lane-root", required=True)

    p = sub.add_parser("rocprof")
    p.add_argument("--lane-root", required=True)
    p.add_argument("--binary", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--surface", required=True, choices=sorted(SURFACE_SHAPES))
    p.add_argument("--out", required=True)
    p.add_argument("--limit", type=int, default=40)
    p.add_argument("--timeout-s", type=int, default=1800)

    p = sub.add_parser("smoke")
    p.add_argument("--lane-root", required=True)
    p.add_argument("--binary", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--timeout-s", type=int, default=600)

    p = sub.add_parser("report")
    for flag in ("--state", "--work-dir", "--store", "--lane-root",
                 "--model27", "--readiness", "--run24-cmd-file"):
        p.add_argument(flag, required=True)

    args = parser.parse_args(argv)
    try:
        return {"probe": cmd_probe, "rocprof": cmd_rocprof,
                "smoke": cmd_smoke, "report": cmd_report}[args.cmd](args)
    except RuntimeError as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
