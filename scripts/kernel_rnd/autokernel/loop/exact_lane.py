"""INF06-EXACT-LANE: the exact-order candidate lane and its token-ID identity judge.

Two acceptance lanes (intake-2674):

* ``tolerance`` (DEFAULT) -- the existing route: quality eval (KLD / op tolerance / AKX
  judges), then speed. Every candidate that does not declare a lane stays here, so
  behaviour is unchanged.
* ``exact`` -- a candidate whose author claims it changes no arithmetic order (loader /
  placement / scheduling / default-off knobs). Its acceptance gate is bitwise identity of
  greedy token IDs versus the incumbent on N fixed prompts x {plain, drafted}, every cell
  in a fresh server process, ignore_eos, fixed seed. PASS skips the quality eval and goes
  straight to speed (A-B-B-A, warm medians). Any mismatch REJECTS the exact claim; the
  caller may re-route the candidate to the tolerance lane (``on_mismatch="tolerance"``),
  but a mismatch is never a silent pass.

The author DECLARES the lane; the loop VERIFIES it with ``judge_streams`` -- a declared
``exact`` lane is only a request to be judged that way.

This is the whole-stream generalisation of ``model_identity.check`` (byte-identical
completions for placement routes): the same greedy-identity idea, but keyed on token IDs
per (prompt, arm) cell, with the drafted arm required, so speculative verification
(N>1 batch shapes) is covered too. The module is stdlib-only so a compute row can carry a
sealed copy of it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
import signal
import subprocess
import time
import urllib.error
import urllib.request
from typing import Iterable, Mapping, Sequence

LANES = ("exact", "tolerance")
DEFAULT_LANE = "tolerance"
ARMS = ("plain", "drafted")
#: Shortest stream that can count as evidence; shorter (or empty) never passes.
DEFAULT_MIN_TOKENS = 32

#: The loop config that distinguishes the two lanes: ordered gate stages per lane.
LANE_GATES: dict[str, tuple[str, ...]] = {
    "tolerance": ("build", "quality_eval", "speed_abba"),
    "exact": ("build", "exact_token_identity", "speed_abba"),
}

ON_MISMATCH = ("reject", "tolerance")


class LaneError(ValueError):
    pass


def candidate_lane(candidate: Mapping | None) -> str:
    """The declared lane of a candidate record; absent/None -> ``tolerance``."""
    lane = (candidate or {}).get("lane")
    if lane is None or lane == "":
        return DEFAULT_LANE
    if lane not in LANES:
        raise LaneError(f"unknown lane {lane!r}; expected one of {LANES}")
    return lane


def gate_plan(candidate: Mapping | None) -> tuple[str, ...]:
    return LANE_GATES[candidate_lane(candidate)]


@dataclass(frozen=True)
class ExactVerdict:
    status: str                      # "pass" | "fail"
    reason: str
    cells: int = 0
    tokens_compared: int = 0
    first_divergence: dict | None = None
    missing: tuple = field(default_factory=tuple)

    @property
    def passed(self) -> bool:
        return self.status == "pass"

    def as_dict(self) -> dict:
        return {"status": self.status, "reason": self.reason, "cells": self.cells,
                "tokens_compared": self.tokens_compared,
                "first_divergence": self.first_divergence,
                "missing": [list(m) for m in self.missing]}


def _ids(stream) -> list[int] | None:
    if not isinstance(stream, (list, tuple)):
        return None
    if not all(isinstance(t, int) and not isinstance(t, bool) for t in stream):
        return None
    return list(stream)


def judge_streams(incumbent: Mapping[tuple[str, str], Sequence[int]],
                  candidate: Mapping[tuple[str, str], Sequence[int]],
                  prompts: Sequence[str], *, arms: Sequence[str] = ARMS,
                  min_tokens: int = DEFAULT_MIN_TOKENS) -> ExactVerdict:
    """PASS iff every (prompt, arm) cell exists on both sides, every stream is a list of
    at least ``min_tokens`` token IDs, and the two streams are identical element-wise.

    No vacuous pass: zero prompts, a missing arm (e.g. no drafted cell), an empty or short
    stream, or non-integer tokens all FAIL."""
    if not prompts:
        return ExactVerdict("fail", "no prompts: identity cannot be vacuous")
    if not arms or "drafted" not in arms or "plain" not in arms:
        return ExactVerdict("fail", "both arms (plain, drafted) are required")
    missing = []
    for p in prompts:
        for a in arms:
            for side, table in (("incumbent", incumbent), ("candidate", candidate)):
                if (p, a) not in table:
                    missing.append((side, p, a))
    if missing:
        return ExactVerdict("fail", f"{len(missing)} cell(s) missing (first: {missing[0]})",
                            missing=tuple(missing))
    compared = 0
    for p in prompts:
        for a in arms:
            inc, cand = _ids(incumbent[(p, a)]), _ids(candidate[(p, a)])
            for side, s in (("incumbent", inc), ("candidate", cand)):
                if s is None:
                    return ExactVerdict("fail", f"{side} {p}/{a}: not a token-ID list")
                if len(s) < min_tokens:
                    return ExactVerdict("fail", f"{side} {p}/{a}: {len(s)} tokens < "
                                        f"min {min_tokens} (no vacuous pass)",
                                        first_divergence={"prompt": p, "arm": a,
                                                          "short": side, "len": len(s)})
            n = min(len(inc), len(cand))
            for i in range(n):
                if inc[i] != cand[i]:
                    return ExactVerdict("fail", f"{p}/{a}: token {i} differs "
                                        f"({inc[i]} vs {cand[i]})", tokens_compared=compared + i,
                                        first_divergence={"prompt": p, "arm": a, "index": i,
                                                          "incumbent": inc[i],
                                                          "candidate": cand[i]})
            if len(inc) != len(cand):
                return ExactVerdict("fail", f"{p}/{a}: lengths differ ({len(inc)} vs "
                                    f"{len(cand)})", tokens_compared=compared + n,
                                    first_divergence={"prompt": p, "arm": a, "index": n,
                                                      "length": [len(inc), len(cand)]})
            compared += n
    cells = len(prompts) * len(arms)
    return ExactVerdict("pass", f"{cells} cells, {compared} token IDs bitwise identical",
                        cells=cells, tokens_compared=compared)


def route_after(candidate: Mapping | None, verdict: ExactVerdict | None, *,
                on_mismatch: str = "reject") -> str:
    """Next stage after the lane's acceptance gate.

    tolerance lane -> ``quality_eval`` (unchanged route). exact lane: PASS -> ``speed_abba``
    (quality eval skipped); FAIL -> ``reject`` or, if asked, ``quality_eval`` (re-routed to
    the tolerance lane). A missing verdict on the exact lane is a FAIL, never a pass."""
    if on_mismatch not in ON_MISMATCH:
        raise LaneError(f"on_mismatch must be one of {ON_MISMATCH}")
    if candidate_lane(candidate) == "tolerance":
        return "quality_eval"
    if verdict is not None and verdict.passed:
        return "speed_abba"
    return "quality_eval" if on_mismatch == "tolerance" else "reject"


def hypothesis_lane(hypothesis) -> str:
    """The lane of a loop `Hypothesis` (attribute, may be absent/empty)."""
    return candidate_lane({"lane": getattr(hypothesis, "lane", "") or None})


def lane_gate(gate, exact_judge=None, *, on_mismatch: str = "reject"):
    """Wrap the loop's injected correctness/quality `gate(hypothesis, paths)` with lane dispatch.

    * tolerance (absent lane): the ORIGINAL `gate` is called with the same arguments and its
      result returned as-is -- today's path, unchanged.
    * exact: `exact_judge(hypothesis, paths) -> ExactVerdict` runs INSTEAD of the quality
      gate. PASS -> (True, [exact_token_identity verdict]) and the loop proceeds straight
      to the A/B measurement. FAIL (or no judge configured, or the judge raising) ->
      (False, [refusal]) i.e. reject; with ``on_mismatch="tolerance"`` the candidate is
      instead re-routed through the original quality gate, with the exact refusal
      recorded first. Never a silent pass."""
    if on_mismatch not in ON_MISMATCH:
        raise LaneError(f"on_mismatch must be one of {ON_MISMATCH}")

    def dispatch(hypothesis, paths):
        if hypothesis_lane(hypothesis) == "tolerance":
            return gate(hypothesis, paths)
        from .gates import Verdict
        if exact_judge is None:
            verdict = ExactVerdict("fail", "exact lane declared but no exact judge configured")
        else:
            try:
                verdict = exact_judge(hypothesis, paths)
            except Exception as exc:  # a harness fault is never a pass
                verdict = ExactVerdict("fail", f"exact judge failed: {type(exc).__name__}: {exc}")
        ok = isinstance(verdict, ExactVerdict) and verdict.passed
        record = Verdict("exact_token_identity", ok,
                         verdict.reason if isinstance(verdict, ExactVerdict) else "no verdict",
                         json.dumps(verdict.as_dict()) if isinstance(verdict, ExactVerdict) else "")
        nxt = route_after({"lane": "exact"}, verdict if ok else None, on_mismatch=on_mismatch)
        if nxt == "speed_abba":
            return True, [record]
        if nxt == "quality_eval":
            passed, verdicts = gate(hypothesis, paths)
            return passed, [record, *verdicts]
        return False, [record]

    return dispatch


# ---------------------------------------------------------------- capture (runtime) --

def request_body(prompt: str, *, n_predict: int, seed: int) -> bytes:
    return json.dumps({"prompt": prompt, "n_predict": n_predict, "temperature": 0,
                       "top_k": 1, "seed": seed, "ignore_eos": True,
                       "cache_prompt": False, "return_tokens": True,
                       "stream": False}).encode()


def serve_one(argv: Sequence[str], env: Mapping[str, str], port: int, body: bytes, *,
              boot_timeout_s: int = 900, request_timeout_s: int = 900,
              log_path: str | None = None) -> list[int]:
    """Fresh llama-server process for ONE cell: launch, one /completion, tear down.

    Kills only the process group this function created. Returns the token IDs."""
    log = open(log_path, "ab") if log_path else subprocess.DEVNULL
    proc = subprocess.Popen(list(argv), env=dict(env), stdin=subprocess.DEVNULL,
                            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    try:
        deadline = time.monotonic() + boot_timeout_s
        while True:
            if proc.poll() is not None:
                raise RuntimeError(f"server exited {proc.returncode} during load")
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2):
                    break
            except (urllib.error.URLError, OSError, ValueError):
                if time.monotonic() > deadline:
                    raise RuntimeError("server not healthy within boot timeout")
                time.sleep(2)
        req = urllib.request.Request(f"http://127.0.0.1:{port}/completion", data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=request_timeout_s) as resp:
            out = json.loads(resp.read())
        tokens = out.get("tokens")
        if not isinstance(tokens, list):
            raise RuntimeError("completion carries no token IDs (return_tokens unsupported?)")
        return [int(t) for t in tokens]
    finally:
        if proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                proc.wait(180)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                proc.wait(10)
        if log is not subprocess.DEVNULL:
            log.close()


def cells(prompts: Iterable[str], arms: Sequence[str] = ARMS):
    return [(p, a) for p in prompts for a in arms]


#: The in-repo CPU campaign config (champion C13 CPU store as incumbent). The lane is
#: CPU-only for now: identical GPU runs diverge (C10-GPU-GREEDY-AA-DIVERGENCE, NOT_LOCALIZED),
#: so token-ID identity cannot be a GPU gate yet.
DEFAULT_CPU_CONFIG = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  "exact_lane_cpu_c13.json")
EXACT_LANE_FLAG = "--exact-lane-config"


def refuse_for_gpu(*, cpu_campaign: bool, gpu_campaign: bool) -> None:
    """The exact lane is CPU-only: refuse it for any GPU (or mixed / non-CPU) campaign."""
    if gpu_campaign or not cpu_campaign:
        raise LaneError("the exact lane is CPU-only (GPU greedy runs are not self-identical: "
                        "C10-GPU-GREEDY-AA-DIVERGENCE); refusing --exact-lane-config for a "
                        "non-CPU campaign")


def with_exact_lane(argv: Sequence[str], *, config=None) -> list[str]:
    """Child argv for the serial launcher: a CPU campaign (`--cpu-serving-launch`) gets
    `--exact-lane-config` (the in-repo default unless one is given); a GPU campaign that
    names one is refused; a GPU campaign without one is returned unchanged."""
    argv = list(argv)
    cpu = "--cpu-serving-launch" in argv
    gpu = "--gpu-serving-launch" in argv
    if EXACT_LANE_FLAG in argv:
        refuse_for_gpu(cpu_campaign=cpu, gpu_campaign=gpu)
        return argv
    if cpu and not gpu:
        return [*argv, EXACT_LANE_FLAG, str(config or DEFAULT_CPU_CONFIG)]
    return argv


def load_config(path) -> dict:
    """Campaign exact-lane config (JSON). Required: incumbent_store, model, draft,
    prompts {id: text}, port. Optional: n_predict (64), seed (1), min_tokens (64),
    threads (4), cpus (taskset list), draft_max (8), log_dir."""
    with open(path) as f:
        cfg = json.load(f)
    missing = [k for k in ("incumbent_store", "model", "draft", "prompts", "port")
               if not cfg.get(k)]
    if missing:
        raise LaneError(f"exact-lane config missing {missing}")
    if cfg.get("backend", "cpu") != "cpu":
        raise LaneError("exact-lane config backend must be cpu (the lane is CPU-only)")
    if not isinstance(cfg["prompts"], dict):
        raise LaneError("exact-lane config prompts must be {id: text}")
    return cfg


def server_argv(cfg: Mapping, store: str, arm: str) -> tuple[list[str], dict]:
    """llama-server argv + env for one cell (CPU, one slot, no warmup, own store)."""
    if cfg.get("threads"):
        nth = str(int(cfg["threads"]))
    else:  # the campaign's own CPU set (physical cores first), never the whole host
        aff = sorted(os.sched_getaffinity(0))
        nth = str(len([c for c in aff if c < 96]) or len(aff))
    argv = [f"{store}/bin/llama-server", "-m", cfg["model"], "-t", nth, "-tb", nth,
            "-ngl", "0", "-c", "2048", "-np", "1", "--port", str(cfg["port"]),
            "--host", "127.0.0.1", "--no-warmup"]
    if cfg.get("cpus"):
        argv = ["taskset", "-c", str(cfg["cpus"]), *argv]
    if arm == "drafted":
        argv += ["--spec-type", "draft-simple", "--spec-draft-model", cfg["draft"],
                 "--spec-draft-ngl", "0", "--spec-draft-n-max", str(int(cfg.get("draft_max", 8))),
                 "--spec-draft-n-min", "1", "--spec-draft-threads", nth,
                 "--spec-draft-threads-batch", nth]
    env = {"PATH": "/usr/bin:/bin", "HOME": os.environ.get("HOME", "/tmp"),
           "LD_LIBRARY_PATH": f"{store}/bin", "OMP_NUM_THREADS": nth,
           "HIP_VISIBLE_DEVICES": ""}
    if os.environ.get("FILLER_ID"):
        env["FILLER_ID"] = os.environ["FILLER_ID"]
    return argv, env


def make_exact_judge(cfg: Mapping, *, capture=serve_one):
    """`make_exact_judge(worker) -> judge(hypothesis, paths) -> ExactVerdict`, for
    `pipeline.run_pool(make_exact_judge=...)`.

    The judge captures every prompt x {plain, drafted} cell for the incumbent store and
    for the worker's candidate build (`worker.build_dir`), each in a fresh server process
    (`capture`, default `serve_one`), then decides with `judge_streams`. It is only ever
    called for hypotheses declaring lane "exact" (`lane_gate`); other candidates never
    reach it."""
    prompts = dict(cfg["prompts"])
    n_predict = int(cfg.get("n_predict", 64))
    seed = int(cfg.get("seed", 1))
    min_tokens = int(cfg.get("min_tokens", n_predict))

    def for_worker(worker):
        candidate_store = str(worker.build_dir)

        def judge(hypothesis, paths) -> ExactVerdict:
            streams = {"incumbent": {}, "candidate": {}}
            for pid, text in prompts.items():
                body = request_body(text, n_predict=n_predict, seed=seed)
                for arm in ARMS:
                    for side, store in (("incumbent", cfg["incumbent_store"]),
                                        ("candidate", candidate_store)):
                        argv, env = server_argv(cfg, store, arm)
                        log = (f"{cfg['log_dir']}/{side}-{arm}-{pid}.log"
                               if cfg.get("log_dir") else None)
                        streams[side][(pid, arm)] = capture(argv, env, int(cfg["port"]),
                                                            body, log_path=log)
            return judge_streams(streams["incumbent"], streams["candidate"], list(prompts),
                                 min_tokens=min_tokens)

        return judge

    return for_worker


__all__ = ["LANES", "DEFAULT_LANE", "ARMS", "LANE_GATES", "LaneError", "ExactVerdict",
           "candidate_lane", "gate_plan", "judge_streams", "route_after", "request_body",
           "serve_one", "cells"]
