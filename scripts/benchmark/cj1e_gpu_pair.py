#!/usr/bin/env python3
"""CJ-1e -- GPQA-Diamond-CoT ranking pair on champion `ef81196d5`, one arm at a time.

Recipe: `docs/design/cj1-gpqa-sample-and-cj1e-gpu-pair.md` §2 at research `b1c7dedb`
(executed from a detached worktree at that commit). Arms:
  A  qwen3.8-27b-q8-dflash2  -- recipe `qwen3.8-27b-q8-gpu-dflash2-np4.json`, port 18371
  B  qwen3.6-35b-a3b-q8-mtp  -- recipe `qwen3.6-35b-a3b-q8-gpu-mtp.json`,     port 18372
argv/env are rendered by `serving.Recipe.server_argv()/server_env()` with the one declared
deviation `ctx 16384 -> 49152` (np=4 => 12288/slot). If a server fails to load at np=4/-c 49152
the arm falls back to np=1 / -c 16384 / runner --concurrency 1, and the record says which ran.
Runner: `v7_quality_gate_runner.py` on the n=198 seeded manifest (`cj_gpqa_sample.py`, seed 42),
chat endpoint, enable_thinking false, temp 0.6 / top_p 0.95 / top_k 20, max_tokens 8192,
`--belief-category CANDIDATE` (the SC32 write side). Residency sampled across each server lifetime;
`/props` must report a non-empty chat_template; post-run: reasoning_chars == 0 on every row and no
`<think>` in content. Analysis: accuracy per arm, both-correct fraction, exact two-sided sign test
on discordant pairs.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.request

CJ = Path("/mnt/raid0/llm/worktrees/sub-gpu-runner-cj1e-b1c7dedb")
sys.path.insert(0, str(CJ / "scripts/kernel_rnd"))
from autokernel.loop import claim, residency, serving  # noqa: E402

BUILD = Path("/mnt/raid0/llm/tmp/build-fold-ef81196d5")
SHA = "869effe5f5cda7f72bd78c8ee168a30f5878b3f32558cd0c02cd38a62a77db37"
ARMS = (
    ("qwen3.8-27b-q8-dflash2", "qwen3.8-27b-q8-gpu-dflash2-np4.json", 18371),
    ("qwen3.6-35b-a3b-q8-mtp", "qwen3.6-35b-a3b-q8-gpu-mtp.json", 18372),
)


def wait_healthy(proc, port, timeout=900):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=3) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(3)
    return False


def stop(proc):
    if proc.poll() is not None:
        return "exited"
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(60)
        return "terminated"
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(30)
        return "killed"


def sign_test(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def run_arm(out: Path, manifest: Path, arm: str, recipe_file: str, port: int) -> dict:
    d = out / arm
    d.mkdir(parents=True, exist_ok=True)
    base = serving.Recipe.load(CJ / "artifacts/serving-recipes" / recipe_file)
    info = {"arm": arm, "recipe": recipe_file, "base_recipe_hash": base.recipe_hash, "attempts": []}
    for np_, ctx, conc in ((4, 49152, 4), (1, 16384, 1)):
        recipe = dataclasses.replace(base, np=np_, ctx=ctx)
        argv = recipe.server_argv(BUILD, port)
        env = recipe.server_env(BUILD)
        env.pop("GGML_NOHUGEPAGE_PROCESS", None)
        attempt = {"np": np_, "ctx": ctx, "concurrency": conc, "recipe_hash": recipe.recipe_hash, "argv": argv}
        info["attempts"].append(attempt)
        sampler = residency.Sampler(interval=1.0)
        with sampler, (d / f"server_np{np_}.stderr").open("wb") as err:
            proc = subprocess.Popen(argv, stdout=subprocess.DEVNULL, stderr=err, env=env)
            attempt["pid"] = proc.pid
            try:
                if not wait_healthy(proc, port):
                    attempt["status"] = f"load_failed rc={proc.poll()}"
                    continue
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/props", timeout=10) as r:
                    props = json.loads(r.read())
                attempt["props_chat_template_len"] = len(props.get("chat_template") or "")
                attempt["props_model_path"] = props.get("model_path")
                attempt["vram_after_load"] = residency.vram_bytes()
                if not attempt["props_chat_template_len"]:
                    attempt["status"] = "refused_no_chat_template"
                    break
                cfg = {"recipe": base.name, "ctx_override": ctx, "np": np_}
                cmd = [sys.executable, "scripts/benchmark/v7_quality_gate_runner.py",
                       "--host", "127.0.0.1", "--port", str(port), "--arm", arm,
                       "--suites", "gpqa_diamond_cot", "--questions-in", str(manifest),
                       "--n", "198", "--seed", "42", "--endpoint", "chat", "--no-enable-thinking",
                       "--temperature", "0.6", "--top-p", "0.95", "--top-k", "20",
                       "--max-tokens", "8192", "--concurrency", str(conc), "--repeats", "1",
                       "--kernel", "champion-ef81196d5", "--binary", str(BUILD / "bin/llama-server"),
                       "--models", recipe.model,
                       "--per-question-out", str(d / "per_question.jsonl"), "--output", str(d / "result.json"),
                       "--belief-category", "CANDIDATE", "--belief-config", json.dumps(cfg)]
                env_r = dict(os.environ, HF_HOME="/mnt/raid0/llm/cache/huggingface", RUNNER_REQUEST_TIMEOUT_S="5400")
                t0 = time.time()
                with (d / "runner.log").open("ab") as log:
                    rc = subprocess.run(cmd, cwd=str(CJ), stdout=log, stderr=subprocess.STDOUT, env=env_r).returncode
                attempt.update(runner_rc=rc, wall_s=round(time.time() - t0, 1),
                               server_alive_at_end=proc.poll() is None)
                attempt["status"] = "ok" if rc == 0 else f"runner_rc={rc}"
                break
            finally:
                attempt["teardown"] = stop(proc)
                attempt["residency"] = sampler.proof
    return info


def analyse(out: Path) -> dict:
    per = {}
    for arm, _, _ in ARMS:
        p = out / arm / "per_question.jsonl"
        if not p.exists():
            continue
        rows = {}
        think = reasoning = 0
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            qid = r.get("id") or r.get("question_id")
            rows[qid] = r
            if "<think>" in str(r.get("response") or r.get("content") or ""):
                think += 1
            if (r.get("reasoning_chars") or 0) > 0:
                reasoning += 1
        per[arm] = {"rows": rows, "think_tag_rows": think, "reasoning_rows": reasoning}
    rep = {"arms": {}}
    for arm, v in per.items():
        rows = v["rows"]
        correct = sum(1 for r in rows.values() if r.get("correct") in (True, 1))
        rep["arms"][arm] = {"n": len(rows), "correct": correct, "accuracy": correct / max(1, len(rows)),
                            "think_tag_rows": v["think_tag_rows"], "reasoning_rows": v["reasoning_rows"]}
    if len(per) == 2:
        (a, va), (b, vb) = list(per.items())
        common = sorted(set(va["rows"]) & set(vb["rows"]))
        ca = [va["rows"][k].get("correct") in (True, 1) for k in common]
        cb = [vb["rows"][k].get("correct") in (True, 1) for k in common]
        b_only_a = sum(1 for x, y in zip(ca, cb) if x and not y)
        b_only_b = sum(1 for x, y in zip(ca, cb) if y and not x)
        rep["paired"] = {"n": len(common), "both_correct": sum(1 for x, y in zip(ca, cb) if x and y) / max(1, len(common)),
                         f"only_{a}": b_only_a, f"only_{b}": b_only_b,
                         "sign_test_p_two_sided": sign_test(b_only_a, b_only_b)}
    return rep


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--analyse-only", action="store_true")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if not args.analyse_only:
        sha = hashlib.sha256((BUILD / "bin/llama-server").read_bytes()).hexdigest()
        if sha != SHA:
            print(f"REFUSED: llama-server sha {sha}", file=sys.stderr)
            return 64
        manifest = args.out / "cj1_gpqa_manifest.json"
        if not manifest.exists():
            subprocess.run([sys.executable, "scripts/benchmark/cj_gpqa_sample.py", "--out", str(manifest)],
                           cwd=str(CJ), check=True)
        infos = []
        with claim.hold() as receipt:
            print(f"GPU claim held: {dict(receipt)}", flush=True)
            for arm, recipe_file, port in ARMS:
                if (args.out / arm / "result.json").exists():
                    continue
                print(f"[{time.strftime('%H:%M:%S')}] arm {arm}", flush=True)
                info = run_arm(args.out, manifest, arm, recipe_file, port)
                infos.append(info)
                print(json.dumps({k: v for k, v in info.items() if k != "attempts"}),
                      [(a["np"], a.get("status"), a.get("wall_s")) for a in info["attempts"]], flush=True)
                (args.out / f"{arm}.serving.json").write_text(json.dumps(info, indent=2))
    rep = analyse(args.out)
    (args.out / "report.json").write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
