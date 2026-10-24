#!/usr/bin/env python3
"""CME-7 post-run check: did the local server reuse the append-only prefix? (stdlib only; reads traces, no model)

  python3 -I cache_hit_report.py <traces_dir> [--min-hit 0.95] [--expect-runs 3 --expect-hands 5] [--out report.json]
  [--belief-out DIR --served-model M --recipe R --np N --ctx C --slots S --kernel-build B]  (needs EPYC_ROOT)

CME-5 sizing rule (predeclared 2026-10-21, queue-builders-3): the acceptance number is NOT cached/input over all
calls, because the first call of every run is cold by construction and early prompts are mostly new suffix, so a
short ``quick_test`` would fail 95% by arithmetic alone. The gated number is over the PREFIX-REUSABLE tokens:

    for call k >= 2 of a run:  reusable_k = min(input_k, input_{k-1} + output_{k-1})
    hit_ratio = sum(cached_k) / sum(reusable_k)            gate: >= --min-hit (default 0.95)

The raw ``sum(cached)/sum(input)`` is reported beside it, never instead.

AMENDMENT 2026-10-22 (CME-7 poker smoke, cme7-cache-miss): the rule above counts the previous call's OUTPUT as
reusable. On a hybrid/recurrent model (Qwen3.6-A3B linear-attention layers) the server resumes only from a
checkpoint made while processing the PROMPT, and the chat template re-renders the previous assistant turn without
its ``<think>`` block, so generated tokens are never reusable: even a perfect layout tops out near 0.91 on this
smoke. The gated basis is therefore ``prompt_prefix`` (default): reusable_k = min(input_k, input_{k-1}). The
CME-5 number is still computed and printed (``hit_ratio_reusable``, ``--basis cme5`` to gate on it) and the missed
tokens are split into ``generated_tokens_unreusable`` / ``history_rewind`` / ``prompt_prefix_lost``. A call whose ``cached_input_tokens`` is
missing makes the report INVALID (a null counter is "not captured", never zero). Exit 0 = complete and gate met,
1 = gate missed, 2 = invalid/incomplete (wrong run or hand count, missing counters).
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import subprocess
import sys
from pathlib import Path


def run_calls(trace: dict) -> list[dict]:
    calls = []
    for it in trace.get("interactions") or []:
        for ev in ((it.get("usage") or {}).get("respond") or {}).get("events") or []:
            calls.append(ev)
    return calls


def analyse(traces: list[dict], min_hit: float = 0.95, expect_runs: int | None = None, expect_hands: int | None = None,
            basis: str = "prompt_prefix") -> dict:
    problems, per_run = [], []
    num = den = raw_c = raw_i = 0
    pnum = pden = split_gen = split_rewind = split_lost = 0
    for n, tr in enumerate(traces):
        calls = run_calls(tr)
        hands = len(tr.get("instance_outcomes") or [])
        if expect_hands is not None and hands != expect_hands:
            problems.append(f"run {n}: {hands} completed hands, expected {expect_hands}")
        if not calls:
            problems.append(f"run {n}: no usage events")
            continue
        rc = rr = 0
        for k, ev in enumerate(calls):
            c, i = ev.get("cached_input_tokens"), ev.get("input_tokens")
            if c is None or i is None:
                problems.append(f"run {n} call {k}: cached/input counter not captured")
                continue
            raw_c += c; raw_i += i
            if k >= 1:
                prev = calls[k - 1]
                reusable = min(i, (prev.get("input_tokens") or 0) + (prev.get("output_tokens") or 0))
                num += c; den += reusable; rc += c; rr += reusable
                pin = prev.get("input_tokens") or 0
                pre = min(i, pin)
                pnum += min(c, pre); pden += pre
                if i < pin:  # history was rewound (parse-retry rollback): nothing of the longer prompt is a prefix
                    split_rewind += reusable - c
                else:
                    split_gen += reusable - pre
                    split_lost += pre - min(c, pre)
        per_run.append({"run": n, "calls": len(calls), "hands": hands, "hit_ratio": round(rc / rr, 4) if rr else None})
    if expect_runs is not None and len(traces) != expect_runs:
        problems.append(f"{len(traces)} runs found, expected {expect_runs}")
    ratio = num / den if den else None
    pratio = pnum / pden if pden else None
    gate = pratio if basis == "prompt_prefix" else ratio
    rep = {"schema": "epyc.clbench_cache_hit.v1", "min_hit": min_hit, "hit_ratio_reusable": None if ratio is None else round(ratio, 4),
           "raw_cached_over_input": round(raw_c / raw_i, 4) if raw_i else None, "reusable_tokens": den, "cached_tokens": num,
           "basis": basis, "hit_ratio_prompt_prefix": None if pratio is None else round(pratio, 4),
           "prompt_prefix_tokens": pden, "prompt_prefix_cached_tokens": pnum,
           "missed_split": {"generated_tokens_unreusable": split_gen, "history_rewind": split_rewind, "prompt_prefix_lost": split_lost},
           "per_run": per_run, "problems": problems}
    rep["verdict"] = "INVALID" if problems or gate is None else ("PASS" if gate >= min_hit else "MISS")
    return rep


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("traces_dir")
    ap.add_argument("--min-hit", type=float, default=0.95)
    ap.add_argument("--basis", choices=("prompt_prefix", "cme5"), default="prompt_prefix")
    ap.add_argument("--expect-runs", type=int)
    ap.add_argument("--expect-hands", type=int)
    ap.add_argument("--out")
    ap.add_argument("--belief-out", metavar="DIR", help="also write the belief sidecar clbench_cache_hit.beliefs.jsonl "
                    "into DIR (needs EPYC_ROOT, --out and the identity flags below; a capture failure exits "
                    "belief_capture.EXIT_CAPTURE_FAILED but --out is still written)")
    ap.add_argument("--run-id", help="belief row run id (default: digest of the trace file list)")
    ident = ap.add_argument_group("belief identity (required with --belief-out; nothing is guessed)")
    ident.add_argument("--served-model")
    ident.add_argument("--recipe")
    ident.add_argument("--np", type=int)
    ident.add_argument("--ctx", type=int)
    ident.add_argument("--slots", type=int)
    ident.add_argument("--kernel-build")
    a = ap.parse_args(argv)
    if a.belief_out:
        missing = [f"--{k.replace('_', '-')}" for k in ("served_model", "recipe", "np", "ctx", "slots", "kernel_build")
                   if getattr(a, k) in (None, "")]
        if not a.out:
            missing.append("--out (the sidecar attests the written report)")
        if missing:
            ap.error("--belief-out requires " + ", ".join(missing))
    files = sorted(glob.glob(str(Path(a.traces_dir) / "run_*.json")))
    rep = analyse([json.load(open(f)) for f in files], a.min_hit, a.expect_runs, a.expect_hands, a.basis)
    rep["trace_files"] = files
    text = json.dumps(rep, indent=1)
    (Path(a.out).write_text(text + "\n") if a.out else print(text))
    rc = {"PASS": 0, "MISS": 1, "INVALID": 2}[rep["verdict"]]
    if not a.belief_out:
        return rc
    return _emit_belief(a, rep, files, rc)


def research_sha() -> str:
    """HEAD of the repo holding THIS script; refuses (raises) when it cannot be read."""
    here = Path(__file__).resolve().parent
    try:
        out = subprocess.run(["git", "-C", str(here), "rev-parse", "HEAD"], capture_output=True, text=True, check=True,
                             timeout=30).stdout.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError(f"cannot read the research SHA from git ({exc})") from exc
    if len(out) != 40:
        raise RuntimeError(f"git rev-parse HEAD returned {out!r}")
    return out


def _emit_belief(a, rep: dict, files: list[str], rc: int) -> int:
    """Belief sidecar via the epyc-root writer (EPYC_ROOT). Imported only here: the script stays stdlib-only without the flag."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import belief_capture  # noqa: E402

    log = belief_capture.CaptureLog()

    def go():
        cap = belief_capture.load_capture("clbench_cache_hit")
        identity = {"served_model": a.served_model, "recipe": a.recipe, "np": a.np, "ctx": a.ctx, "slots": a.slots,
                    "kernel_build": a.kernel_build, "research_sha": research_sha(), "basis": rep["basis"]}
        run_id = a.run_id or "clbench-" + hashlib.sha256("\n".join(files).encode()).hexdigest()[:16]
        Path(a.belief_out).mkdir(parents=True, exist_ok=True)
        return cap.write_belief_measurements(a.belief_out, run_id=run_id, producer="cache_hit_report.py",
                                             report_path=a.out, identity=identity)

    side = log.run("clbench_cache_hit", go)
    if "clbench_cache_hit" in log.written:
        print(f"belief sidecar: {side}", file=sys.stderr)
    return log.exit_code(rc)


if __name__ == "__main__":
    sys.exit(main())
