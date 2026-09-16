#!/usr/bin/env python3
"""PRB-T4 -- the PRB-T2 TALE-EP evaluation on architect_general's model, served on the MI210.

Handoff: `handoffs/active/per-request-reasoning-budget.md` (PRB-T2 design, PRB-T4 run).

SERVER. The production `architect_general` launch, resolved at run time from the orchestrator's
own builder (`orchestrator_stack.build_server_command(..., prepare_runtime_dirs=False)` +
`stack_numa._numa_prefix`) -- never transcribed. Two deltas, both stated in the record:
the port (not :8083, which is production's) and `--slot-save-path` (a scratch dir, so this run
never writes production's slot directory). Frozen v9 HIP binary, run only.
GPU residency is sampled across the whole window; the GPU claim is held.

HARNESS. `scripts/benchmark/eval_tale_budget.py` at research `a454b7fd` (a detached worktree;
the question pool is the shared clone's untracked `benchmarks/prompts/question_pool.jsonl`).
Each suite runs separately at the handoff's own n (math 150, olympiadbench 100, mmlu_pro 150,
livecodebench 100), all three arms, `--budget-unit tokens`, temperature = architect_general's
declared `generation_defaults.temperature` (0.1, `orchestration/model_registry.yaml`), seed 42,
`chat_template_kwargs {"enable_thinking": false}`. The CPU `frontdoor` replicate is dropped
(no CPU inference in this window).

ANALYSIS. Per suite: accuracy and mean tokens per arm; paired bootstrap (questions resampled,
10,000 draws, seed 20260916) 95% CIs for the accuracy delta (pp) and the NET token change
(TALE counts its estimator call); then the handoff's fixed decision rule, verbatim.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import signal
import statistics
import subprocess
import sys
import time
import urllib.request

HERE = Path(__file__).resolve()
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "scripts/kernel_rnd"))
from autokernel.loop import claim, residency  # noqa: E402

ORCH = Path("/workspace/repos/epyc-orchestrator")
TALE_WT = Path("/mnt/raid0/llm/worktrees/sub-gpu-runner-tale-a454b7fd")
PORT = 18383
SUITES = {"math": 150, "olympiadbench": 100, "mmlu_pro": 150, "livecodebench": 100}
TEMPERATURE = 0.1  # model_registry.yaml roles.architect_general.generation_defaults.temperature
GGUF = "/mnt/raid0/llm/models/Qwen3.8-27B-Q8_0.gguf"
CONDITIONS = ("baseline", "static", "tale")

RESOLVE = r"""
import json, sys
sys.path.insert(0, 'scripts/server'); sys.path.insert(0, '.')
import orchestrator_stack as o
from stack_numa import _numa_prefix
reg = o.RegistryLoader()
rc = reg.get_role_config('architect_general') if hasattr(reg, 'get_role_config') else reg.roles['architect_general']
print(json.dumps({'prefix': _numa_prefix('architect_general'),
                  'cmd': o.build_server_command(rc, %d, prepare_runtime_dirs=False)}))
"""


def resolve_argv(slot_dir: Path) -> tuple[list[str], dict]:
    out = subprocess.run([str(ORCH / ".venv/bin/python3"), "-c", RESOLVE % PORT], cwd=str(ORCH),
                         capture_output=True, text=True, check=True, timeout=120)
    spec = json.loads(out.stdout.strip().splitlines()[-1])
    cmd = list(spec["cmd"])
    i = cmd.index("--slot-save-path")
    production_slot_dir = cmd[i + 1]
    cmd[i + 1] = str(slot_dir)
    return spec["prefix"] + cmd, {"production_slot_save_path": production_slot_dir,
                                  "resolved_prefix": spec["prefix"], "resolved_cmd": spec["cmd"]}


def load_rows(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            if "question_id" in r:
                rows.append(r)
    return rows


def boot_ci(values_by_q: dict, stat, draws=10000, seed=20260916):
    keys = sorted(values_by_q)
    rng = random.Random(seed)
    point = stat([values_by_q[k] for k in keys])
    samples = []
    for _ in range(draws):
        pick = [values_by_q[keys[rng.randrange(len(keys))]] for _ in keys]
        samples.append(stat(pick))
    samples.sort()
    return {"point": point, "lo": samples[int(0.025 * draws)], "hi": samples[int(0.975 * draws) - 1]}


def analyse(rows: list[dict]) -> dict:
    by = {}
    for r in rows:
        by.setdefault(r["suite"], {}).setdefault(r["condition"], {})[r["question_id"]] = r
    out = {}
    for suite, arms in sorted(by.items()):
        base = arms.get("baseline", {})
        s = {"arms": {}}
        for cond, qs in arms.items():
            scored = [q for q in qs.values() if q.get("correct") is not None]
            s["arms"][cond] = {
                "n": len(qs), "n_scored": len(scored),
                "accuracy": (sum(bool(q["correct"]) for q in scored) / len(scored)) if scored else None,
                "mean_answer_tokens": statistics.mean(q["total_tokens"] for q in qs.values()),
                "mean_tokens_incl_estimator": statistics.mean(
                    q.get("total_tokens_incl_estimator") or q["total_tokens"] for q in qs.values()),
                "mean_latency_s_incl_estimator": statistics.mean(
                    q.get("elapsed_s_incl_estimator") or q["elapsed_s"] for q in qs.values()),
                "mean_estimator_tokens": statistics.mean(q.get("estimator_tokens") or 0 for q in qs.values()),
            }
            if cond == "tale":
                budgets = sorted(q["tale_budget"] for q in qs.values() if q.get("tale_budget") is not None)
                if budgets:
                    s["arms"][cond]["budget_quartiles"] = [budgets[0], budgets[len(budgets) // 4],
                                                           budgets[len(budgets) // 2],
                                                           budgets[3 * len(budgets) // 4], budgets[-1]]
            if cond == "baseline":
                continue
            common = [k for k in qs if k in base and qs[k].get("correct") is not None
                      and base[k].get("correct") is not None]
            pairs = {k: (float(bool(qs[k]["correct"])), float(bool(base[k]["correct"])),
                         float(qs[k].get("total_tokens_incl_estimator") or qs[k]["total_tokens"]),
                         float(base[k]["total_tokens"])) for k in common}
            if not pairs:
                continue
            s["arms"][cond]["acc_delta_pp"] = boot_ci(
                pairs, lambda v: 100.0 * (sum(x[0] for x in v) - sum(x[1] for x in v)) / len(v))
            s["arms"][cond]["net_token_reduction_pct"] = boot_ci(
                pairs, lambda v: 100.0 * (1 - sum(x[2] for x in v) / max(1.0, sum(x[3] for x in v))))
            s["arms"][cond]["n_paired"] = len(pairs)
        out[suite] = s
    return out


def decide(summary: dict) -> dict:
    """The PRB-T2 fixed decision rule, applied to point estimates; CIs are reported beside it."""
    suites = list(summary)
    tale = {s: summary[s]["arms"].get("tale", {}) for s in suites}
    static = {s: summary[s]["arms"].get("static", {}) for s in suites}
    red = {s: tale[s].get("net_token_reduction_pct", {}).get("point") for s in suites}
    dacc = {s: tale[s].get("acc_delta_pp", {}).get("point") for s in suites}
    good = [s for s in suites if red[s] is not None and red[s] >= 30 and dacc[s] is not None and dacc[s] >= -1.0]
    worst = min((v for v in dacc.values() if v is not None), default=None)

    def static_wins(s):
        sr = static[s].get("net_token_reduction_pct", {}).get("point")
        sa = static[s].get("acc_delta_pp", {}).get("point")
        return sr is not None and sa is not None and sr >= (red[s] or 0) and sa >= (dacc[s] or 0)

    tale_beats_static = all(not static_wins(s) for s in suites)
    mean_red = statistics.mean(v for v in red.values() if v is not None) if any(v is not None for v in red.values()) else None
    if len(good) >= 3 and worst is not None and worst >= -3.0 and tale_beats_static:
        verdict = "ADOPT"
    elif all(static_wins(s) for s in suites):
        verdict = "ADOPT_STATIC_INSTEAD"
    elif (worst is not None and worst < -3.0) or (mean_red is not None and mean_red < 15):
        verdict = "DECLINE"
    else:
        verdict = "INCONCLUSIVE"
    return {"verdict": verdict, "suites_meeting_30pct_and_minus1pp": good,
            "worst_suite_acc_delta_pp": worst, "tale_beats_static_everywhere": tale_beats_static,
            "static_wins_by_suite": {s: static_wins(s) for s in suites},
            "tale_net_reduction_pct_by_suite": red, "tale_acc_delta_pp_by_suite": dacc,
            "mean_tale_net_reduction_pct": mean_red,
            "note": "DECLINE's '<15% net reduction' is read on the mean across suites; per-suite values are above."}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--suites", nargs="+", default=list(SUITES))
    ap.add_argument("--analyse-only", action="store_true")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    if not args.analyse_only:
        slot_dir = args.out / "slots"
        slot_dir.mkdir(exist_ok=True)
        argv, provenance = resolve_argv(slot_dir)
        import hashlib
        binary_sha = hashlib.sha256(Path(argv[argv.index("-m") - 1]).read_bytes()).hexdigest()
        provenance["binary_sha256"] = binary_sha
        env = dict(os.environ)
        env.pop("HSA_OVERRIDE_GFX_VERSION", None)
        env.pop("GGML_NOHUGEPAGE_PROCESS", None)
        env["LD_LIBRARY_PATH"] = "/mnt/raid0/llm/llama.cpp/build-hip/bin:/opt/rocm/lib"
        (args.out / "server.json").write_text(json.dumps({"argv": argv, **provenance,
                                                           "LD_LIBRARY_PATH": env["LD_LIBRARY_PATH"]}, indent=2))
        sampler = residency.Sampler(interval=1.0)
        with claim.hold() as receipt, sampler, (args.out / "server.stderr").open("ab") as err:
            print(f"GPU claim held: {dict(receipt)}", flush=True)
            proc = subprocess.Popen(argv, stdout=subprocess.DEVNULL, stderr=err, env=env)
            print(f"server pid {proc.pid}: {' '.join(argv)}", flush=True)
            try:
                deadline = time.time() + 900
                while True:
                    if proc.poll() is not None:
                        raise RuntimeError(f"server exited rc={proc.returncode}")
                    try:
                        with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=3) as r:
                            if r.status == 200:
                                break
                    except Exception:
                        pass
                    if time.time() > deadline:
                        raise RuntimeError("server not healthy in 900s")
                    time.sleep(3)
                vram = residency.vram_bytes()
                print(f"healthy; VRAM {vram / 2**30:.1f} GiB, KFD {residency.kfd_processes()}", flush=True)
                if vram < 8 * 2**30:
                    raise RuntimeError(f"VRAM {vram} below 8 GiB -- not GPU-resident")
                for suite in args.suites:
                    outp = args.out / f"prb_t4_gpu_{suite}_{ts}.jsonl"
                    cmd = [sys.executable, "scripts/benchmark/eval_tale_budget.py",
                           "--endpoint", f"http://127.0.0.1:{PORT}", "--suites", suite,
                           "--n-questions", str(SUITES[suite]), "--conditions", *CONDITIONS,
                           "--budget-unit", "tokens", "--temperature", str(TEMPERATURE), "--seed", "42",
                           "--chat-template-kwargs", '{"enable_thinking": false}', "--output", str(outp)]
                    t0 = time.time()
                    print(f"[{time.strftime('%H:%M:%S')}] suite {suite} n={SUITES[suite]}", flush=True)
                    with (args.out / f"{suite}.log").open("ab") as log:
                        rc = subprocess.run(cmd, cwd=str(TALE_WT), stdout=log, stderr=subprocess.STDOUT).returncode
                    print(f"  rc={rc} [{time.time() - t0:.0f}s] vram_now={residency.vram_bytes() / 2**30:.1f}G "
                          f"peak={sampler.peak_vram / 2**30:.1f}G", flush=True)
                    meta_path = outp.with_suffix(".meta.json")
                    if rc == 0 and outp.exists() and meta_path.exists():
                        served = json.loads(meta_path.read_text()).get("serving") or {}
                        ok_gguf = (served.get("gguf_path") == GGUF and served.get("gguf_path_source") == "props")
                        print(f"  served gguf {served.get('gguf_path')} via {served.get('gguf_path_source')}: "
                              f"{'OK' if ok_gguf else 'MISMATCH'}", flush=True)
                        # Belief kernel, write side (VB-PRB-T4): emitted at write time, never backfilled.
                        try:
                            sys.path.insert(0, "/workspace/scripts/vidya")
                            from adapters import tale_budget_capture as cap
                            side = cap.write_belief_measurements(
                                outp, run_id=f"prb_t4_gpu_{ts}_{suite}",
                                producer="epyc-inference-research/scripts/benchmark/prb_t4_tale_gpu.py",
                                kernel={"binary_path": argv[argv.index("-m") - 1],
                                        "binary_sha256": binary_sha,
                                        "tree": "production-consolidated-v9"})
                            print(f"  beliefs: {side}", flush=True)
                        except Exception as exc:
                            print(f"  beliefs REFUSED: {type(exc).__name__}: {exc}", flush=True)
                    if proc.poll() is not None:
                        raise RuntimeError(f"server died during {suite}")
            finally:
                if proc.poll() is None:
                    proc.send_signal(signal.SIGTERM)
                    try:
                        proc.wait(60)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait(30)
                print(f"server {proc.pid} stopped rc={proc.returncode}", flush=True)
        (args.out / "residency.json").write_text(json.dumps(sampler.proof, indent=2))
    rows = []
    metas = {}
    for p in sorted(args.out.glob("prb_t4_gpu_*.jsonl")):
        rows += load_rows(p)
        meta = Path(str(p) + ".meta.json")
        if not meta.exists():
            meta = p.with_suffix(".meta.json")
        if meta.exists():
            metas[p.name] = json.loads(meta.read_text())
    summary = analyse(rows)
    report = {"schema": "epyc.prb_t4.tale_gpu_report.v1", "temperature": TEMPERATURE, "seed": 42,
              "budget_unit": "tokens", "suites_n": SUITES, "per_suite": summary,
              "decision": decide(summary) if summary else None,
              "serving_identity": {k: v.get("serving") for k, v in metas.items()}}
    (args.out / "report.json").write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report["decision"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
