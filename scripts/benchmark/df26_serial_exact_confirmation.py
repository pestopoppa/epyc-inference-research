#!/usr/bin/env python3
"""DF2-6 confirmation -- is the greedy non-parity a 1-row vs multi-row numeric split?

Spec: `/workspace/progress/2026-09/2026-09-16-sub-akfix.md` -> "Minimal GPU confirmation experiment".
Hypothesis: DF2-6's losslessness failures come from the plain-decode (1-row) vs verify-batch
(multi-row) numeric split (MMVQ vs MMQ, GDN chunked vs autoregressive, FA vec vs rocWMMA), not
from a DFlash2 defect.

Binary: `build-champion-c463f601b-hip-20260909` (carries `LLAMA_SPEC_EXACT` and
`LLAMA_SPEC_DIAG_ACCEPT`; descends from `ef81196d5`). Replays the DF2-6 server command
(`df2_greedy_parity.arm_argv`: -np 1, f16 KV, -fa on, n-max 8, host threads 184-191) with the
same 12 prompts, temp 0 / top_k 1 / seed 42, 256 tokens, a fresh process per arm.

Arms: none_A, none_B (A/A); draft-simple and draft-dflash each with LLAMA_SPEC_EXACT unset
(+ GGML_CUDA_LOG_MMVQ_ROUTE=2, LLAMA_SPEC_DIAG_ACCEPT=1) and with LLAMA_SPEC_EXACT=serial.
The serial mode is confirmed from the server's own startup line.

PASS rule (from the spec): A/A 12/12 identical; serial arms 12/12 with draft_n > 0 on every
prompt; unset arms reproduce failures with a small top-2 margin at the first differing index.
A failing serial arm means the drafter corrupts target state, not batch shape.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

HERE = Path(__file__).resolve()
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "scripts/kernel_rnd"))
sys.path.insert(0, str(HERE.parent))
from autokernel.loop import claim, residency  # noqa: E402
import df2_greedy_parity as df2  # noqa: E402

BUILD_BIN = Path("/mnt/raid0/llm/tmp/build-champion-c463f601b-hip-20260909/bin")
PIN = "184-191"
DIAG_ENV = {"GGML_CUDA_LOG_MMVQ_ROUTE": "2", "LLAMA_SPEC_DIAG_ACCEPT": "1"}
ARMS = (
    ("none_A", "baseline", {}),
    ("simple_unset", "draft_simple", DIAG_ENV),
    ("dflash_unset", "dflash2", DIAG_ENV),
    ("simple_serial", "draft_simple", {"LLAMA_SPEC_EXACT": "serial"}),
    ("dflash_serial", "dflash2", {"LLAMA_SPEC_EXACT": "serial"}),
    ("none_B", "baseline", {}),
)
DIAG_RE = re.compile(r"\[spec_diag\] role=(\w+) row=(\d+) idx=(-?\d+) n_rows=(\d+) tgt_sampled=(-?\d+) "
                     r"tgt_argmax=(-?\d+) tgt_argmax_logit=(\S+) top2=(-?\d+) margin=(\S+) draft=(-?\d+)")
MODE_RE = re.compile(r"speculative exactness mode \(LLAMA_SPEC_EXACT\) = (\w+)")


def run_arm(out: Path, label: str, kind: str, extra_env: dict, prompts: list, max_tokens: int, ctx: int) -> dict:
    d = out / label
    d.mkdir(parents=True, exist_ok=True)
    log = d / "server.stderr"
    argv = df2.arm_argv(BUILD_BIN, kind, ctx, PIN)
    (d / "server_command.txt").write_text(" ".join(argv) + "\n")
    env = dict(os.environ)
    env.pop("HSA_OVERRIDE_GFX_VERSION", None)
    env.pop("GGML_NOHUGEPAGE_PROCESS", None)
    for key in ("LLAMA_SPEC_EXACT", "GGML_CUDA_LOG_MMVQ_ROUTE", "LLAMA_SPEC_DIAG_ACCEPT"):
        env.pop(key, None)
    env["LD_LIBRARY_PATH"] = f"{BUILD_BIN}:/opt/rocm/lib"
    env.update(extra_env)
    (d / "env.json").write_text(json.dumps({k: env[k] for k in ("LD_LIBRARY_PATH", *extra_env)}, indent=2))
    records = []
    sampler = residency.Sampler(interval=0.5)
    with sampler, log.open("wb") as errf:
        proc = subprocess.Popen(argv, stdout=errf, stderr=subprocess.STDOUT, env=env)
        try:
            df2.wait_ready(proc, log)
            vram = df2.read_vram()
            if vram < df2.VRAM_RESIDENT_FLOOR:
                raise df2.ArmRefused(f"{label}: VRAM {vram} below residency floor")
            for q in prompts:
                start = log.stat().st_size
                resp = df2.ask(q["prompt"], max_tokens)
                time.sleep(0.3)
                end = log.stat().st_size
                t = resp.get("timings") or {}
                records.append({"id": q["id"], "tokens": resp.get("tokens") or [],
                                "content_sha256": hashlib.sha256((resp.get("content") or "").strip().encode()).hexdigest(),
                                "draft_n": t.get("draft_n"), "draft_n_accepted": t.get("draft_n_accepted"),
                                "predicted_n": t.get("predicted_n"), "log_span": [start, end]})
        finally:
            teardown_pid = proc.pid
            df2.stop(proc)
    text = log.read_bytes()
    for rec in records:
        seg = text[rec["log_span"][0]:rec["log_span"][1]].decode(errors="replace")
        rec["diag_rows"] = [dict(zip(("role", "row", "idx", "n_rows", "sampled", "argmax", "argmax_logit",
                                      "top2", "margin", "draft"), m.groups())) for m in DIAG_RE.finditer(seg)]
    full = text.decode(errors="replace")
    modes = MODE_RE.findall(full)
    (d / "records.json").write_text(json.dumps(records, indent=1))
    return {"label": label, "kind": kind, "env": extra_env, "records": records, "vram_bytes": vram,
            "server_pid": teardown_pid, "spec_exact_mode_logged": modes,
            "route_lines": full.count("MUL_MAT_ROUTE") + full.count("mmvq_route"),
            "residency": sampler.proof}


def margin_at(rec: dict, k: int) -> dict:
    """The arm's own verification row that produced generation token k (token 0 is the prompt pass)."""
    rows = rec.get("diag_rows") or []
    toks = rec["tokens"]
    sampled = [int(r["sampled"]) for r in rows]
    n = min(len(sampled), len(toks) - 1)
    mapped = n > 0 and sampled[:n] == toks[1:1 + n]
    if not mapped or k < 1 or k - 1 >= len(rows):
        return {"mapped": False, "diag_rows": len(rows)}
    r = rows[k - 1]
    return {"mapped": True, "role": r["role"], "row": int(r["row"]), "n_rows": int(r["n_rows"]),
            "sampled": int(r["sampled"]), "top2": int(r["top2"]), "margin": float(r["margin"])}


def compare(a: dict, b: dict, with_margin: bool = False) -> dict:
    rows = []
    for rec in a["records"]:
        ref = next((x for x in b["records"] if x["id"] == rec["id"]), None)
        if ref is None:
            continue
        same = rec["tokens"] == ref["tokens"]
        row = {"id": rec["id"], "verdict": "PASS" if same else "FAIL",
               "first_diff": None if same else df2.first_diff_index(ref["tokens"], rec["tokens"]),
               "draft_n": rec["draft_n"], "draft_n_accepted": rec["draft_n_accepted"]}
        if with_margin and not same and row["first_diff"] is not None:
            k = row["first_diff"]
            m = margin_at(rec, k)
            if m.get("mapped") and k < len(ref["tokens"]):
                m["reference_token"] = ref["tokens"][k]
                m["reference_is_top2"] = ref["tokens"][k] == m["top2"]
            row["at_first_diff"] = m
        rows.append(row)
    return {"n_pass": sum(r["verdict"] == "PASS" for r in rows), "n": len(rows),
            "all_drafted": all((r["draft_n"] or 0) > 0 for r in rows), "per_prompt": rows}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--ctx", type=int, default=32768)
    args = ap.parse_args()
    prompts = json.loads(df2.QUESTIONS.read_text())[:12]
    args.out.mkdir(parents=True, exist_ok=True)
    sha = hashlib.sha256((BUILD_BIN / "llama-server").read_bytes()).hexdigest()
    results = {}
    with claim.hold() as receipt:
        print(f"GPU claim held: {dict(receipt)}; llama-server {sha[:16]}", flush=True)
        for label, kind, env in ARMS:
            print(f"[{time.strftime('%H:%M:%S')}] {label}", flush=True)
            try:
                results[label] = run_arm(args.out, label, kind, env, prompts, args.max_tokens, args.ctx)
                print(f"  mode={results[label]['spec_exact_mode_logged']} route_lines={results[label]['route_lines']} "
                      f"vram={results[label]['vram_bytes']/2**30:.1f}G", flush=True)
            except Exception as exc:
                print(f"  REFUSED {type(exc).__name__}: {exc}", flush=True)
                results[label] = {"label": label, "error": f"{type(exc).__name__}: {exc}"}
    ok = {k: v for k, v in results.items() if "records" in v}
    report = {"schema": "epyc.inf62.df26_serial_exact.v1", "build_bin": str(BUILD_BIN),
              "llama_server_sha256": sha, "prompts": [p["id"] for p in prompts],
              "max_tokens": args.max_tokens, "arms": {k: {x: v.get(x) for x in (
                  "kind", "env", "spec_exact_mode_logged", "route_lines", "vram_bytes", "residency", "error")}
                  for k, v in results.items()}, "comparisons": {}}
    if "none_A" in ok and "none_B" in ok:
        report["comparisons"]["none_B~none_A"] = compare(ok["none_B"], ok["none_A"])
    for label in ("simple_unset", "dflash_unset", "simple_serial", "dflash_serial"):
        if label in ok and "none_A" in ok:
            report["comparisons"][f"{label}~none_A"] = compare(ok[label], ok["none_A"], with_margin=True)
    c = report["comparisons"]
    aa = c.get("none_B~none_A", {})
    serial_ok = all(c.get(f"{l}~none_A", {}).get("n_pass") == 12 and c[f"{l}~none_A"]["all_drafted"]
                    and ok[l]["spec_exact_mode_logged"] == ["serial"] for l in ("simple_serial", "dflash_serial")
                    if f"{l}~none_A" in c) and all(f"{l}~none_A" in c for l in ("simple_serial", "dflash_serial"))
    unset_fail = [c[f"{l}~none_A"]["n"] - c[f"{l}~none_A"]["n_pass"] for l in ("simple_unset", "dflash_unset")
                  if f"{l}~none_A" in c]
    report["verdict"] = {"aa_identical": aa.get("n_pass") == 12,
                         "serial_arms_12_of_12_drafted_and_mode_confirmed": serial_ok,
                         "unset_arm_failures": unset_fail,
                         "hypothesis_supported": aa.get("n_pass") == 12 and serial_ok and any(unset_fail)}
    (args.out / "report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report["verdict"], indent=1))
    for key, cmp_ in c.items():
        fails = [(r["id"][-4:], r["first_diff"], (r.get("at_first_diff") or {}).get("margin"))
                 for r in cmp_["per_prompt"] if r["verdict"] == "FAIL"]
        print(f"  {key:22s} {cmp_['n_pass']}/{cmp_['n']} drafted_all={cmp_['all_drafted']} fails={fails}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
