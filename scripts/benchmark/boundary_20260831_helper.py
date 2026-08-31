#!/usr/bin/env python3
"""Helper for boundary_20260831.sh — the run-21→22 overnight boundary driver.

Three subcommands, all invoked by the driver (never run these by hand mid-boundary):

  verdict <json>   Print DIVERGENCE / PARITY / UNREADABLE from the funsafe-math
                   admission report. The driver's merge conditional consumes this.

  merge ...        The operator-ratified flag-removal merge: tag the champion tip,
                   cherry-pick the admission commit onto it, rebuild, run the
                   test-backend-ops MUL_MAT oracle under the device claim, and
                   ROLL BACK to the tag on ANY failure. Exit 0 only on a fully
                   verified merge.

  report ...       Write the run-22 readiness package (no device). Includes the
                   exact run-22 launch command and the PENDING line — run starts
                   are operator-gated; NOTHING in this file launches run 22.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent


def _git(tree: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(tree), *args],
                          capture_output=True, text=True, timeout=600)


def _git_ok(tree: Path, *args: str) -> str:
    done = _git(tree, *args)
    if done.returncode != 0:
        raise RuntimeError(f"git -C {tree} {' '.join(args)}: {done.stderr.strip()}")
    return done.stdout.strip()


# --------------------------------------------------------------------- verdict
def cmd_verdict(args: argparse.Namespace) -> int:
    """DIVERGENCE iff the harness demonstrated >=1 greedy argmax split on gfx90a."""
    try:
        report = json.loads(Path(args.json_path).read_text(encoding="utf-8"))
        diverged = int(report["divergent_prompts"])
    except Exception:
        print("UNREADABLE")
        return 0
    print("DIVERGENCE" if diverged > 0 else "PARITY")
    return 0


# ----------------------------------------------------------------------- merge
def cmd_merge(args: argparse.Namespace) -> int:
    champ, tree = Path(args.champ_tree), Path(args.admission_tree)
    build_dir = champ / "build-hip"

    tip = _git_ok(champ, "rev-parse", f"refs/heads/{args.branch}")
    head = _git_ok(champ, "rev-parse", "HEAD")
    if head != tip:
        print(f"REFUSED: {champ} HEAD {head[:12]} is not {args.branch} tip {tip[:12]}")
        return 1
    if _git_ok(champ, "status", "--porcelain"):
        print(f"REFUSED: {champ} working tree dirty — will not cherry-pick into it")
        return 1

    admission = _git_ok(tree, "rev-parse", args.admission_ref)
    # One-line geometry re-check at merge time (the harness checked its own copy).
    parent = _git_ok(tree, "rev-parse", f"{args.admission_ref}^")
    touched = _git_ok(tree, "diff", "--name-only", parent, admission).splitlines()
    if touched != ["ggml/src/ggml-hip/CMakeLists.txt"]:
        print(f"REFUSED: admission commit touches {touched}, not exactly the CMake line")
        return 1

    # Tag first: the rollback anchor AND the audit trail of what tip we merged onto.
    if _git(champ, "rev-parse", "--verify", f"refs/tags/{args.tag}").returncode == 0:
        print(f"REFUSED: tag {args.tag} already exists — a previous merge attempt "
              "left state behind; operator review needed")
        return 1
    _git_ok(champ, "tag", args.tag, tip)
    print(f"tagged    {args.tag} = {tip[:12]}")

    def rollback(reason: str) -> int:
        print(f"ROLLBACK: {reason} — resetting {args.branch} to {args.tag}")
        _git(champ, "cherry-pick", "--abort")
        _git(champ, "reset", "--hard", args.tag)
        return 1

    pick = _git(champ, "cherry-pick", admission)
    if pick.returncode != 0:
        return rollback(f"cherry-pick of {admission[:12]} failed: {pick.stderr.strip()[-300:]}")
    merged = _git_ok(champ, "rev-parse", "HEAD")
    print(f"picked    {admission[:12]} onto {tip[:12]} -> {merged[:12]}")

    # Rebuild on the build cores (96-183, per the loop's own serial-build policy);
    # no claim needed to compile (run 9's idle-while-claimed defect).
    build = subprocess.run(["taskset", "-c", "96-183", "cmake", "--build",
                            str(build_dir), "-j", str(args.build_jobs)],
                           capture_output=True, text=True)
    if build.returncode != 0:
        return rollback(f"rebuild failed rc={build.returncode}: "
                        f"{(build.stderr or build.stdout)[-300:]}")
    print("rebuilt   champion build-hip at merged tip")

    # Oracle under the loop's own claim (device work): positive-evidence MUL_MAT
    # correctness via gates.op_correctness — the same oracle the loop trusts.
    sys.path.insert(0, str(HERE.parent / "kernel_rnd"))
    from autokernel.loop import claim, gates  # noqa: PLC0415
    try:
        with claim.hold() as receipt:
            print(f"claim     held on {receipt['device_id']}")
            verdict = gates.op_correctness(build_dir)
    except Exception as exc:  # claim refused, oracle crashed — either way: rollback
        return rollback(f"oracle could not run: {exc}")
    if not verdict.passed:
        return rollback(f"MUL_MAT oracle failed: {verdict.reason}")
    print(f"oracle    MUL_MAT passed on merged build")
    print(f"MERGED    {args.branch} @ {merged}")
    return 0


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
    """Every value ever written for `key`, in order (reasons accumulate)."""
    if not path.is_file():
        return []
    return [line.partition("=")[2] for line in path.read_text(encoding="utf-8").splitlines()
            if line.startswith(f"{key}=")]


def _floors(store: Path) -> dict[str, str]:
    rows = {}
    for surface in ("dec-b2", "dec-b4", "dec-b8"):
        path = store / "calibration" / f"{surface}.json"
        if path.is_file():
            try:
                floor = json.loads(path.read_text(encoding="utf-8"))["floor_pct"]
                rows[surface] = (", ".join(f"{k}p: {float(v):.3f}%"
                                           for k, v in sorted(floor.items(), key=lambda kv: int(kv[0])))
                                 + f"  ({path})")
            except Exception as exc:
                rows[surface] = f"UNREADABLE ({path}: {exc})"
        else:
            rows[surface] = f"NOT CALIBRATED ({path} missing)"
    return rows


def cmd_report(args: argparse.Namespace) -> int:
    state = _state(Path(args.state))
    store = Path(args.store)
    work = Path(args.work_dir)
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    tip = _git(Path(args.champ_tree), "rev-parse", f"refs/heads/{args.branch}")
    tip = tip.stdout.strip() if tip.returncode == 0 else "UNRESOLVED"
    merged = state.get("step1_merged", "unknown")
    verdict = state.get("step1_verdict", "unknown")

    # Serving bundle: prefer the dated bundle of this boundary, else the canonical.
    refresh_date = state.get("refresh_date", "")
    bundle_path = Path(args.surface_dir) / (f"operator_gate_bundle_{refresh_date}.json"
                                            if refresh_date else "operator_gate_bundle.json")
    headline = "UNAVAILABLE (bundle missing or unreadable)"
    if bundle_path.is_file():
        try:
            bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
            head = bundle.get("headline") or {}
            headline = head.get("summary", "bundle sealed, no headline field")
        except Exception as exc:
            headline = f"UNREADABLE ({exc})"

    floors = _floors(store)
    steps = []
    for step, label in (("step0", "stop run 21"), ("step1", "funsafe-math admission A/B"),
                        ("step2_dec-b2", "calibrate dec-b2"), ("step2_dec-b4", "calibrate dec-b4"),
                        ("step2_dec-b8", "calibrate dec-b8"), ("step3", "serving evidence refresh"),):
        done = state.get(step) == "done"
        extra = state.get(f"{step}_outcome", "")
        steps.append(f"| {label} | {'DONE' if done else 'NOT DONE'} "
                     f"| {extra or '-'} | {work}/{step.replace('step2_', 'step2-').replace('step0', 'step0-stop').replace('step1', 'step1-funsafe').replace('step3', 'step3-refresh')}.log |")

    anchor_line = (
        f"`{args.champ_tree}/build-hip` (merged champion build; oracled in step 1; "
        f"hand-built — run 22 via `--allow-unverified-anchor`, OR promote a fresh "
        f"anchor-gen first for an attested one). NOTE: `{args.anchor_gen}` carries "
        f"provenance for the PRE-merge tip; verify_anchor's ancestor rule still "
        f"accepts it, one keep behind."
        if merged == "yes" else
        f"`{args.anchor_gen}` (provenance.json champion_commit = pre-boundary tip; "
        f"ancestor-or-equal of HEAD — passes `champion.verify_anchor`)")
    anchor_arg = (f"{args.champ_tree}/build-hip"
                  if merged == "yes" else str(args.anchor_gen))
    waiver = " \\\n      --allow-unverified-anchor" if merged == "yes" else ""

    allgreen = state.get("allgreen", "unknown")
    reasons = _state_all(Path(args.state), "allgreen_reason")
    if allgreen == "yes":
        allgreen_block = (
            "**ALL GREEN.** Every boundary step met its gate. The OP-32 "
            "pre-authorization condition is satisfied — but this driver does not "
            "start runs (the unattended start was denied by the permission system "
            "at build time; run starts stay operator-gated here). The command "
            "below is verified-ready: paste it to launch run 22.")
    elif allgreen == "no":
        allgreen_block = ("**NOT all green — HOLD.** Run 22 must NOT start. Red steps:\n\n"
                          + "\n".join(f"- {r}" for r in reasons))
    else:
        allgreen_block = "**UNEVALUATED — HOLD.** The driver never reached the all-green check."

    prominent_parity = ""
    if verdict == "PARITY":
        prominent_parity = (
            "\n> **PROMINENT — FLAG VERDICT: PARITY.** The funsafe-math harness found "
            "ZERO greedy divergence on gfx90a at these shapes: the quality gain is "
            "UNDEMONSTRATED on our silicon, the operator ruling's condition (\"IF it "
            "increases quality as stated\") is NOT met, and **nothing was merged**. "
            "The champion is unchanged. Whether upstream parity alone is worth the "
            "measured decode cost is an open operator decision "
            f"(evidence: `{args.funsafe_json}`).\n")

    report = f"""# Run-21 → Run-22 boundary report — 2026-08-31 overnight

Generated {now} by `scripts/benchmark/boundary_20260831.sh` (unattended).
Driver log: `{work}/driver.log` · state: `{args.state}`

## Per-step outcomes

| Step | State | Outcome | Log |
|---|---|---|---|
{chr(10).join(steps)}

Run 21: pid {state.get('run21_pid', '?')} — {state.get('step0_outcome', '?')}, \
escalation: {state.get('step0_escalated', 'n/a')}, \
group remnants: {state.get('step0_group_remnants', 'n/a')}.

## Flag verdict (`-funsafe-math-optimizations` removal, CH-7)
{prominent_parity}
- Harness verdict: **{verdict}** (rc={state.get('step1_harness_rc', '?')}; \
evidence `{args.funsafe_json}`)
- Merged onto champion: **{merged}** \
{'(pre-merge tag `ak/pre-funsafe-merge-20260831`)' if merged in ('yes', 'rolled_back') else ''}
- Admission re-cut: {state.get('step1_recut', 'n/a')}
- Champion tip now: `{args.branch}` @ `{tip[:12]}` \
(pre-boundary: `{state.get('champion_tip_pre_boundary', '?')[:12]}`)

## Calibration floors (as written to the store)

- dec-b2: {floors['dec-b2']}
- dec-b4: {floors['dec-b4']}
- dec-b8: {floors['dec-b8']}

## Serving evidence bundle

- Headline: {headline}
- Path: `{bundle_path}` (canonical: `{args.surface_dir}/operator_gate_bundle.json`)

## OP-32 all-green verdict (operator ruling 2026-08-31: "pre-authorize run 22 if all boundary steps are green")

{allgreen_block}

## Run-22 launch command (DO NOT RUN without the operator's go)

Anchor: {anchor_line}

```bash
cd {args.lane_root}/scripts/kernel_rnd && \\
  setsid nohup python3 -u -m autokernel.loop.run \\
      --worktree {args.champ_tree} \\
      --anchor-build {anchor_arg}{waiver} \\
      --model {args.model} \\
      --store {store} \\
      --iterations 0 --surface tg128 --pairs 20 --workers 7 \\
      --rank-prior-experiments \\
      --worker-root /mnt/raid0/llm/tmp/ak-lanes \\
      --worker-build-root /mnt/raid0/llm/tmp/ak-lane-builds \\
      --out {store}/run22 \\
      > /mnt/raid0/llm/tmp/run22.launch.log 2>&1 &
echo $! > /mnt/raid0/llm/tmp/run22.pid
```

dec-b2/b4/b8 are now calibrated and available as `--surface` values, but run 22's
PRIMARY surface stays **tg128** unless the operator says otherwise.

## PENDING

**Run 22 awaits the operator's explicit go — run starts are operator-gated and
this driver did NOT start it.** Under OP-32, an ALL-GREEN verdict above means
the pre-authorization condition is met and pasting the launch command IS the
go; any HOLD verdict means run 22 must not start until the red steps are
resolved.
"""
    Path(args.report).write_text(report, encoding="utf-8")
    print(f"wrote {args.report}")
    return 0


# ------------------------------------------------------------------------ main
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_verdict = sub.add_parser("verdict")
    p_verdict.add_argument("json_path")

    p_merge = sub.add_parser("merge")
    p_merge.add_argument("--champ-tree", required=True)
    p_merge.add_argument("--branch", required=True)
    p_merge.add_argument("--admission-tree", required=True)
    p_merge.add_argument("--admission-ref", required=True)
    p_merge.add_argument("--tag", required=True)
    p_merge.add_argument("--build-jobs", type=int, default=64)

    p_report = sub.add_parser("report")
    for flag in ("--state", "--work-dir", "--store", "--surface-dir", "--report",
                 "--champ-tree", "--branch", "--model", "--lane-root",
                 "--anchor-gen", "--funsafe-json"):
        p_report.add_argument(flag, required=True)

    args = parser.parse_args(argv)
    try:
        return {"verdict": cmd_verdict, "merge": cmd_merge,
                "report": cmd_report}[args.cmd](args)
    except RuntimeError as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
