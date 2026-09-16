#!/usr/bin/env python3
"""INF-62 SL-1 -- DFlash2 `--spec-draft-n-max` x `--spec-draft-p-min` sweep on the champion.

Handoff: `handoffs/active/dflash2-block-drafter-experimental-build.md` -> SL-1.
Grid: n-max {4,6,7,8} x p-min {0,0.5}, in-flight {1,2,4,8}, >=5 interleaved rounds,
temperature 0.6 / seed 42. Reports `aggregate_tok_s` AND verifier steps/s.

HARNESS. Every sample is ONE `autokernel.loop.serving.calibrate_floor(..., samples=1)`
launch of the canonical recipe (`artifacts/serving-recipes/qwen3.8-27b-q8-gpu-dflash2-np4.json`)
with only the swept fields replaced -- the same codified path as the 2026-09-08
maximum-performance sweep. It keeps the harness's own residency proof (a launch measured
non-resident is refused) and its warmup round. In-flight == `np`, because the harness fires
exactly `np` concurrent requests per round.

WORKLOAD DELTAS FROM THE CANONICAL RECIPE (all stated in each row's recipe_hash):
  temperature 0.6 (recipe default top_k 20, top_p 0.95), `--seed 42` as the server default
  sampling seed, `--spec-draft-p-min P` (always passed, including 0.00, so the arms differ
  in the value only), `spec_decode.draft_n_max`.

VERIFIER STEPS. The harness keeps only `predicted_n`/`predicted_per_second` per request.
This runner tees each `/completion` response (after the harness has read it -- no effect
on the timed path) and keeps `timings.draft_n` / `draft_n_accepted` / `predicted_ms`.
A DFlash verification step emits accepted+1 tokens and the first token comes from the
prompt pass, so   steps = predicted_n - 1 - draft_n_accepted   (exact when every later
token came from a verification step; SL-2 lands the server's own counter in serving.py).
verifier_steps_s = sum over slots of steps / (predicted_ms/1000) -- the same
sum-of-per-slot-rates construction as `aggregate_tok_s`.

SOURCE FACTS THAT SHAPE THE READING (champion `ef81196d5`, common/speculative.cpp):
  * dflash.block_size = 8 in the drafter GGUF, n_draft_max = block_size-1 = 7, and a
    larger n_max is CLAMPED (:1390-1395). n-max 8 == n-max 7.
  * the `is_dflash2` draft branch (:1657-1700) never reads `params.p_min`; p_min is only
    consulted by the DFlash1 / DSpark / other drafter branches. p-min 0.5 is expected to
    be a no-op here. Measured, not assumed: the grid carries both.
So the 8 arms are 3 distinct configurations, and the equivalent arms are A/A controls.

GPU claim: `autokernel.loop.claim.hold()` for the whole run (refuses if held).
Results: one JSON line per launch, appended as it completes (resumable).
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
from pathlib import Path
import random
import statistics
import sys
import threading
import time
import urllib.request

HERE = Path(__file__).resolve()
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "scripts/kernel_rnd"))
from autokernel.loop import claim, serving  # noqa: E402

BUILD = Path("/mnt/raid0/llm/tmp/build-fold-ef81196d5")
BUILD_SERVER_SHA256 = "869effe5f5cda7f72bd78c8ee168a30f5878b3f32558cd0c02cd38a62a77db37"
RECIPE = REPO / "artifacts/serving-recipes/qwen3.8-27b-q8-gpu-dflash2-np4.json"
N_MAX = (4, 6, 7, 8)
P_MIN = (0.0, 0.5)
IN_FLIGHT = (1, 2, 4, 8)
SEED = 42
TEMPERATURE = 0.6

# ---------------------------------------------------------------------------------------
# response tee (post-read; the harness has already stopped its clock for this request)
# ---------------------------------------------------------------------------------------
_SINK: list[dict] = []
_SINK_LOCK = threading.Lock()
_real_urlopen = urllib.request.urlopen


class _Tee:
    def __init__(self, inner, t0):
        self._inner, self._t0 = inner, t0

    def read(self, *args):
        data = self._inner.read(*args)
        t1 = time.monotonic()
        try:
            timings = json.loads(data).get("timings", {})
        except Exception:  # the harness reports its own parse failure
            timings = {}
        with _SINK_LOCK:
            _SINK.append({"t_start": self._t0, "t_end": t1,
                          **{k: timings.get(k) for k in (
                              "predicted_n", "predicted_ms", "predicted_per_second",
                              "draft_n", "draft_n_accepted", "prompt_n")}})
        return data

    def __getattr__(self, name):
        return getattr(self._inner, name)


def _urlopen(req, *args, **kwargs):
    url = req.full_url if isinstance(req, urllib.request.Request) else str(req)
    t0 = time.monotonic()
    resp = _real_urlopen(req, *args, **kwargs)
    return _Tee(resp, t0) if url.endswith("/completion") else resp


urllib.request.urlopen = _urlopen


def arm_recipe(base: serving.Recipe, n_max: int, p_min: float, np_: int) -> serving.Recipe:
    sd = dict(base.spec_decode)
    sd["draft_n_max"] = n_max
    flags = tuple(base.extra_flags) + ("--seed", str(SEED), "--spec-draft-p-min", f"{p_min:.2f}")
    return dataclasses.replace(
        base, name=f"{base.name}-sl1-n{n_max}-p{int(p_min * 100):02d}-np{np_}",
        spec_decode=sd, np=np_, temperature=TEMPERATURE,
        top_k=serving.Recipe.__dataclass_fields__["top_k"].default,
        top_p=serving.Recipe.__dataclass_fields__["top_p"].default,
        extra_flags=flags)


def arm_id(n_max, p_min, np_):
    return f"n{n_max}-p{int(p_min * 100):02d}-np{np_}"


def schedule(rounds: int, seed: int) -> list[tuple]:
    """Interleaved: every round visits every (n_max, p_min) arm at every in-flight level;
    arm order is reshuffled per (round, level) so no arm always follows the same one."""
    rng = random.Random(seed)
    arms = [(n, p) for n in N_MAX for p in P_MIN]
    plan = []
    for r in range(rounds):
        for np_ in IN_FLIGHT:
            order = arms[:]
            rng.shuffle(order)
            plan += [(r, n, p, np_) for n, p in order]
    return plan


def summarize(rows: list[dict]) -> dict:
    out = {}
    ok = [r for r in rows if r.get("status") == "ok"]
    for key in sorted({r["arm"] for r in ok}):
        sel = [r for r in ok if r["arm"] == key]
        tok = [r["aggregate_tok_s"] for r in sel]
        stp = [r["verifier_steps_s"] for r in sel]
        acc = [r["tokens_per_step"] for r in sel]
        sp = serving._spread(tok) if len(tok) >= 2 else {}
        out[key] = {"n": len(sel),
                    "median_tok_s": statistics.median(tok),
                    "min_tok_s": min(tok), "max_tok_s": max(tok),
                    "p95_dev_pct": sp.get("p95_dev_pct"), "cv_pct": sp.get("cv_pct"),
                    "median_verifier_steps_s": statistics.median(stp),
                    "min_steps_s": min(stp), "max_steps_s": max(stp),
                    "median_tokens_per_step": statistics.median(acc),
                    "failures": sum(1 for r in rows if r["arm"] == key and r.get("status") != "ok")}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True, help="results directory (JSONL appended)")
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--plan-seed", type=int, default=20260916)
    ap.add_argument("--port", type=int, default=18361)
    ap.add_argument("--execute", action="store_true")
    args = ap.parse_args(argv)

    if "GGML_NOHUGEPAGE_PROCESS" in os.environ:
        print("REFUSED: GGML_NOHUGEPAGE_PROCESS is set; the GPU serving recipe runs without it", file=sys.stderr)
        return 64
    import hashlib
    sha = hashlib.sha256((BUILD / "bin/llama-server").read_bytes()).hexdigest()
    if sha != BUILD_SERVER_SHA256:
        print(f"REFUSED: llama-server sha256 {sha} != champion {BUILD_SERVER_SHA256}", file=sys.stderr)
        return 64
    base = serving.Recipe.load(RECIPE)
    plan = schedule(args.rounds, args.plan_seed)
    args.out.mkdir(parents=True, exist_ok=True)
    rows_path = args.out / "launches.jsonl"
    done = set()
    rows = []
    if rows_path.exists():
        for line in rows_path.read_text().splitlines():
            r = json.loads(line)
            rows.append(r)
            if r.get("status") == "ok":
                done.add((r["round"], r["arm"]))
    print(f"plan {len(plan)} launches, {len(done)} already done; build {BUILD} ({sha[:12]})", flush=True)
    if not args.execute:
        for step in plan[:12]:
            r = arm_recipe(base, step[1], step[2], step[3])
            print(step, r.describe())
        return 0
    meta = {"schema": "epyc.inf62.sl1_sweep.v1", "build": str(BUILD), "llama_server_sha256": sha,
            "champion_commit": "ef81196d5bdd4190b46dff4ae7eecc333a46c8ce",
            "base_recipe": str(RECIPE), "base_recipe_hash": base.recipe_hash,
            "grid": {"n_max": N_MAX, "p_min": P_MIN, "in_flight": IN_FLIGHT, "rounds": args.rounds},
            "temperature": TEMPERATURE, "seed": SEED, "plan_seed": args.plan_seed,
            "unit": serving.CALIBRATION_UNIT, "harness": "serving.calibrate_floor(samples=1) per launch",
            "steps_estimator": "predicted_n - 1 - draft_n_accepted per slot",
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    (args.out / "meta.json").write_text(json.dumps(meta, indent=2))
    with claim.hold() as receipt:
        print(f"GPU claim held: {dict(receipt)}", flush=True)
        for rnd, n_max, p_min, np_ in plan:
            key = arm_id(n_max, p_min, np_)
            if (rnd, key) in done:
                continue
            recipe = arm_recipe(base, n_max, p_min, np_)
            with _SINK_LOCK:
                _SINK.clear()
            t0 = time.time()
            row = {"round": rnd, "arm": key, "n_max": n_max, "p_min": p_min, "in_flight": np_,
                   "recipe_hash": recipe.recipe_hash, "argv": recipe.server_argv(BUILD, args.port),
                   "started_at": t0}
            try:
                res = serving.calibrate_floor(recipe, BUILD, samples=1, port=args.port)
                with _SINK_LOCK:
                    tees = list(_SINK)
                meas = tees[np_:]  # first np completions = the harness's warmup round
                if len(meas) != np_:
                    raise RuntimeError(f"tee captured {len(tees)} completions, expected {2 * np_}")
                steps = []
                for t in meas:
                    s = t["predicted_n"] - 1 - (t["draft_n_accepted"] or 0)
                    steps.append(s)
                    t["verif_steps_est"] = s
                tok_sum = sum(t["predicted_per_second"] for t in meas)
                agg = res["runs"][0]
                if abs(tok_sum - agg) > 1e-6 * max(1.0, agg):
                    raise RuntimeError(f"tee/harness mismatch: {tok_sum} vs {agg}")
                steps_s = sum(s / (t["predicted_ms"] / 1000.0) for s, t in zip(steps, meas))
                tokens = sum(t["predicted_n"] for t in meas)
                row.update(status="ok", aggregate_tok_s=agg, verifier_steps_s=steps_s,
                           tokens_per_step=(tokens - np_) / max(1, sum(steps)),
                           draft_n=sum(t["draft_n"] or 0 for t in meas),
                           draft_n_accepted=sum(t["draft_n_accepted"] or 0 for t in meas),
                           predicted_n=tokens,
                           wall_round_s=max(t["t_end"] for t in meas) - min(t["t_start"] for t in meas),
                           slots=meas, residency=res["residency"],
                           launch_residency=res["launch_residency"])
            except Exception as exc:  # recorded, never smoothed; the plan continues
                row.update(status="failed", error=f"{type(exc).__name__}: {exc}"[:500],
                           record=getattr(exc, "record", None))
            row["seconds"] = round(time.time() - t0, 1)
            rows.append(row)
            with rows_path.open("a") as fh:
                fh.write(json.dumps(row, default=str) + "\n")
            if row["status"] == "ok":
                print(f"r{rnd} {key:14s} agg {row['aggregate_tok_s']:8.2f} tok/s  steps/s "
                      f"{row['verifier_steps_s']:7.2f}  tok/step {row['tokens_per_step']:.3f}  "
                      f"res {row['residency'].get('status')}  [{row['seconds']}s]", flush=True)
            else:
                print(f"r{rnd} {key:14s} FAILED {row['error']}", flush=True)
            (args.out / "summary.json").write_text(json.dumps(summarize(rows), indent=2))
    print("done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
