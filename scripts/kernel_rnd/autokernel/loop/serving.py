#!/usr/bin/env python3
"""Serving-throughput measurement: a keep is real only if it improves SERVING, not a
llama-bench proxy (R23-43, operator directive 2026-09-04: "the only performance that
matters is serving performance").

WHY THIS EXISTS. llama-bench measures a fixed-workload forward pass. It is deterministic
and cheap -- the right tool for the planner to SCREEN hypotheses -- but it is a PROXY, and
2026-09-04 proved the proxy diverges: two keeps worth +23.3% / +10.1% on the dec-b4 bench
surface moved DFlash2 serving decode by ~0% (71.22 t/s, flat). So the KEEP GATE and the
HEADLINE move to `llama-server` under the champion's CANONICAL RECIPE, which is also the
recipe production needs at promotion -- built once, used for both.

THE RECIPE IS GENERAL. `spec_decode.type` is one of {none, draft-dflash, draft-mtp, ...}; a
model that does not use speculative decode carries `none` and its own optimal `np`. Nothing
about DFlash2 is baked into the framework; today's champion just happens to serve the 27B on
gfx90a with DFlash2 at np4 (the aggregate-throughput knee measured by DF2-5).

THE METRIC. `aggregate_tok_s` = sum of predicted tokens across `np` concurrent requests /
wall time -- what a busy server sustains, which is what the operator chose to optimise.

This module is backend-blind about the kernel: it takes two BUILD DIRECTORIES (champion vs
candidate) and the recipe, and returns a paired A/B `Comparison`-shaped result the loop's
keep gate already knows how to read.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import concurrent.futures as cf
import json
import os
from pathlib import Path
import statistics
import subprocess
import time
import urllib.request

#: Prompts fired at the server. Distinct so the slots do not share a KV prefix (a shared
#: prefix would understate the real per-request work); enough of them to cover np up to 8.
_PROMPTS = (
    "Prove by induction that the sum of the first n odd numbers is n squared, then compute n=20.",
    "Explain how a red-black tree keeps its height logarithmic, then insert 7,3,18,10,22,8,11.",
    "Derive the closed form of the Fibonacci sequence via generating functions, step by step.",
    "A train leaves A at 60 km/h and another leaves B at 90 km/h 200 km apart; when do they meet? Show work.",
    "Prove that there are infinitely many primes, then list the first ten primes above 1000.",
    "Explain the CAP theorem and give a concrete example system for each of CP, AP, and CA.",
    "Compute the eigenvalues of [[2,1],[1,2]] and explain what they mean geometrically.",
    "Describe Dijkstra's algorithm and trace it on a 6-node weighted graph you define.",
)


@dataclass(frozen=True)
class Recipe:
    """A champion's canonical serving configuration. General over spec-decode type."""
    name: str
    model: str
    device: str = "ROCm0"
    ngl: int = 99
    #: {"type": "none"} | {"type": "draft-dflash"|"draft-mtp", "drafter": path, "ngld": int,
    #: "draft_n_max": int}
    spec_decode: dict = field(default_factory=lambda: {"type": "none"})
    np: int = 4
    ctx: int = 16384
    threads: int = 8
    batch: int = 2048
    ubatch: int = 2048
    ctk: str = "f16"
    ctv: str = "f16"
    fa: str = "on"
    kv_unified: bool = False
    extra_flags: tuple = ()
    #: workload
    n_predict: int = 256
    temperature: float = 0.6
    top_p: float = 0.95
    top_k: int = 20
    metric: str = "aggregate_tok_s"

    @classmethod
    def load(cls, path: Path | str) -> "Recipe":
        d = json.loads(Path(path).read_text())
        d.pop("schema", None)
        d["extra_flags"] = tuple(d.get("extra_flags", ()))
        return cls(**d)

    def server_argv(self, build_dir: Path, port: int) -> list[str]:
        argv = [str(Path(build_dir) / "bin" / "llama-server"),
                "-m", self.model, "-np", str(self.np), "-c", str(self.ctx),
                "-t", str(self.threads), "-tb", str(self.threads),
                "-b", str(self.batch), "-ub", str(self.ubatch),
                "-ctk", self.ctk, "-ctv", self.ctv, "--device", self.device,
                "-ngl", str(self.ngl), "-fa", self.fa,
                "--host", "127.0.0.1", "--port", str(port), "--metrics", "--slots"]
        sd = self.spec_decode
        if sd.get("type", "none") != "none":
            argv += ["-md", sd["drafter"], "-ngld", str(sd.get("ngld", self.ngl)),
                     "--spec-type", sd["type"]]
            if "draft_n_max" in sd:
                argv += ["--spec-draft-n-max", str(sd["draft_n_max"])]
        argv += ["--kv-unified" if self.kv_unified else "--no-kv-unified"]
        argv += list(self.extra_flags)
        return argv

    def describe(self) -> str:
        sd = self.spec_decode.get("type", "none")
        return f"{self.name} [np{self.np} {sd} {self.metric}]"


class ServerDied(RuntimeError):
    """The server exited during load or measurement -- a build/config fault, not noise."""


def _measure_once(recipe: Recipe, build_dir: Path, port: int,
                  boot_timeout_s: int = 360) -> float:
    """Launch the server under `recipe`, fire `np` concurrent requests, return aggregate
    tok/s. The server is always stopped, even on error."""
    argv = recipe.server_argv(build_dir, port)
    env = dict(os.environ, LD_LIBRARY_PATH=str(Path(build_dir) / "bin"))
    srv = subprocess.Popen(argv, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env)
    try:
        for _ in range(boot_timeout_s // 2):
            if srv.poll() is not None:
                raise ServerDied(f"server exited {srv.returncode} during load ({recipe.describe()})")
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2)
                break
            except Exception:
                time.sleep(2)
        else:
            raise ServerDied("server not healthy within boot timeout")

        def one(i: int) -> int:
            body = json.dumps({"prompt": _PROMPTS[i % len(_PROMPTS)],
                               "n_predict": recipe.n_predict, "temperature": recipe.temperature,
                               "top_p": recipe.top_p, "top_k": recipe.top_k,
                               "cache_prompt": False}).encode()
            req = urllib.request.Request(f"http://127.0.0.1:{port}/completion", data=body,
                                         headers={"Content-Type": "application/json"})
            d = json.loads(urllib.request.urlopen(req, timeout=600).read())
            return int(d.get("timings", {}).get("predicted_n", 0))

        t0 = time.time()
        with cf.ThreadPoolExecutor(recipe.np) as ex:
            toks = list(ex.map(one, range(recipe.np)))
        wall = time.time() - t0
        if min(toks) < recipe.n_predict // 2 or wall <= 0:
            raise ServerDied(f"degenerate measurement: tokens={toks} wall={wall:.2f}")
        return sum(toks) / wall
    finally:
        srv.terminate()
        try:
            srv.wait(30)
        except Exception:
            srv.kill()
            srv.wait(10)


def compare(recipe: Recipe, anchor_build: Path, candidate_build: Path, *, pairs: int,
            floor_pct: float | None, port: int = 18311) -> dict:
    """Paired, alternating serving A/B: anchor vs candidate, `pairs` times, each pair a
    fresh server per side (drift control). Effect = median(candidate)/median(anchor) - 1.
    `decisive` is None when uncalibrated (no floor), so the keep gate fails closed."""
    a_runs, c_runs = [], []
    for _ in range(pairs):
        a_runs.append(_measure_once(recipe, anchor_build, port))
        c_runs.append(_measure_once(recipe, candidate_build, port))
    a_med, c_med = statistics.median(a_runs), statistics.median(c_runs)
    effect = c_med / a_med - 1.0
    decisive = None if floor_pct is None else (abs(effect) * 100.0 >= floor_pct)
    return {"schema": "epyc.autokernel.serving_ab.v1", "recipe": recipe.name,
            "metric": recipe.metric, "np": recipe.np, "pairs": pairs,
            "anchor_tok_s": a_med, "candidate_tok_s": c_med,
            "effect": effect, "effect_pct": effect * 100.0,
            "noise_floor_pct": floor_pct, "decisive": decisive,
            "anchor_samples": a_runs, "candidate_samples": c_runs}


def calibrate_floor(recipe: Recipe, build_dir: Path, *, samples: int, port: int = 18311) -> dict:
    """A/A the serving metric `samples` times on ONE build: the run-to-run spread IS the
    noise floor a keep must clear. floor = p95 of |pairwise effect| against the median,
    reported at a few sample counts so a keep at N pairs is judged against the N-pair bar."""
    runs = [_measure_once(recipe, build_dir, port) for _ in range(samples)]
    med = statistics.median(runs)
    devs = sorted(abs(r / med - 1.0) * 100.0 for r in runs)
    p95 = devs[min(len(devs) - 1, int(round(0.95 * (len(devs) - 1))))]
    return {"schema": "epyc.autokernel.serving_floor.v1", "recipe": recipe.name,
            "metric": recipe.metric, "np": recipe.np, "samples": samples,
            "median_tok_s": med, "floor_pct": round(p95, 3),
            "runs": runs, "cv_pct": round(statistics.pstdev(runs) / med * 100.0, 3)}


__all__ = ["Recipe", "ServerDied", "calibrate_floor", "compare"]
