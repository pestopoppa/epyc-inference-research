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

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
import concurrent.futures as cf
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import time
import urllib.request

#: Schema of the on-disk recipe file, and part of the recipe's identity: a schema bump
#: changes what the fields MEAN, so it must change the hash too.
RECIPE_SCHEMA = "epyc.autokernel.canonical_recipe.v1"

#: Variables the LOADER owns. A recipe may not set these, and the refusal is a hard error
#: rather than a silent override: `LD_LIBRARY_PATH` is what pins the build's OWN ggml, and
#: three ggml generations live on this host -- a binary that inherits another tree's ggml
#: runs silently wrong and no exit code reports it. `HSA_OVERRIDE_GFX_VERSION` is
#: deliberately UNSET by the loader env for the same reason. An arm that needs either of
#: these is not an env arm; it is a different build or a different device.
LOADER_OWNED_ENV = ("LD_LIBRARY_PATH", "HSA_OVERRIDE_GFX_VERSION")

#: `env_readback` key standing for "the recipe does not set this variable at all".
UNSET = "unset"

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
    #: CPU affinity for the server's host threads, e.g. "184-191". None = UNPINNED, which is
    #: what every floor calibrated to date measured, so it stays the default: setting it
    #: changes the measured condition and INVALIDATES the recipe's serving floor until that
    #: floor is re-calibrated (R23-49). Unpinned host threads are free to land on 0-95, the
    #: CPU campaign's bench region -- kernel-verified 2026-09-07, every logical CPU on this
    #: host shares a physical core with 0-95, so this pollutes their arms and our own.
    cpu_list: str | None = None
    #: workload
    n_predict: int = 256
    temperature: float = 0.6
    top_p: float = 0.95
    top_k: int = 20
    metric: str = "aggregate_tok_s"
    #: Extra environment variables set on the server process at LAUNCH, on top of the
    #: loader env. This is what makes a launch-time knob expressible as an ARM at all:
    #: `GGML_NOHUGEPAGE_PROCESS=1` (a `prctl(PR_SET_THP_DISABLE)` shim, distinct from the
    #: already-on madvise `GGML_NOHUGEPAGE`) cannot be reached through argv, and on the CPU
    #: surface it cut between-launch variance 25.3x (sd 2.510% -> 0.481%). Default None
    #: keeps every shipped recipe launching EXACTLY as it does today: this makes the arm
    #: expressible, it does not adopt it. Setting it CHANGES THE MEASURED CONDITION and so
    #: invalidates the recipe's serving floor until that floor is re-calibrated, exactly as
    #: `cpu_list` does (R23-49).
    env: dict | None = None
    #: Declarative post-launch readback: prove the env arm TOOK EFFECT. Each entry is
    #: {"field": <'/proc/<pid>/status' field>, "expect": <value>} for an unconditional
    #: check, or {"field": ..., "env": <variable>, "expect": {<env value>: <field value>,
    #: "unset": <field value>}} for one that asserts in BOTH directions from a single
    #: declaration -- e.g.
    #:     {"field": "THP_enabled", "env": "GGML_NOHUGEPAGE_PROCESS",
    #:      "expect": {"1": "0", "unset": "1"}}
    #: reads THP_enabled=0 on the treatment arm and THP_enabled=1 on the control. An env
    #: state the declaration does not cover is a REFUSAL at construction, not a skipped
    #: check: a control arm whose readback was never declared is a control that was never
    #: checked, and a control that is secretly the treatment cannot be detected afterwards.
    env_readback: tuple = ()

    def __post_init__(self) -> None:
        for key, value in (self.env or {}).items():
            if not isinstance(key, str) or not isinstance(value, str):
                raise RecipeError(
                    f"env {key!r}={value!r}: environment variables are strings. Quote the "
                    f"value in the recipe JSON -- coercing it here would let the record "
                    f"claim a value the process never saw.")
            if key in LOADER_OWNED_ENV:
                raise RecipeError(
                    f"recipe env may not set {key!r}: it is owned by the loader env, and "
                    f"overriding it would break the three-ggml-generations linkage "
                    f"guarantee the residency check depends on. An arm that needs a "
                    f"different {key} is a different BUILD, not an env arm.")
        # Resolve the readback declaration NOW, so an uncovered env state fails at
        # construction (including through `with_env`) rather than mid-measurement.
        self.readback_expectations()

    @classmethod
    def load(cls, path: Path | str) -> "Recipe":
        return cls.from_dict(json.loads(Path(path).read_text()))

    @classmethod
    def from_dict(cls, d: Mapping) -> "Recipe":
        d = dict(d)
        d.pop("schema", None)
        d["extra_flags"] = tuple(d.get("extra_flags", ()))
        if d.get("env") is not None:
            d["env"] = dict(d["env"])
        d["env_readback"] = tuple(dict(c) for c in (d.get("env_readback") or ()))
        return cls(**d)

    def to_dict(self) -> dict:
        """Every field, in the shape `from_dict` reads back. There are no cosmetic fields:
        each one either changes what is launched or changes what the number means."""
        return {"schema": RECIPE_SCHEMA, "name": self.name, "model": self.model,
                "device": self.device, "ngl": self.ngl,
                "spec_decode": dict(self.spec_decode), "np": self.np, "ctx": self.ctx,
                "threads": self.threads, "batch": self.batch, "ubatch": self.ubatch,
                "ctk": self.ctk, "ctv": self.ctv, "fa": self.fa,
                "kv_unified": self.kv_unified, "extra_flags": list(self.extra_flags),
                "cpu_list": self.cpu_list, "n_predict": self.n_predict,
                "temperature": self.temperature, "top_p": self.top_p, "top_k": self.top_k,
                "metric": self.metric, "env": dict(self.env or {}),
                "env_readback": [dict(c) for c in self.env_readback]}

    @property
    def recipe_hash(self) -> str:
        """Content-addressed identity of the SERVING recipe, emitted with every serving row.

        A champion identified only by a commit hash is under-specified once a launch recipe
        is part of the artifact (R23-59): the same commit launched at a different `np`, a
        different pin, or -- now that `env` exists -- with a different launch-time knob is a
        different measurement wearing the same name. This digest is what lets a gate REFUSE
        a measurement whose recipe does not match the one the floor was calibrated under,
        instead of silently comparing two conditions.

        It covers `to_dict()`, i.e. every field including `env` and `env_readback`. `None`
        and `{}` env normalise to the same digest, because "no extra env" is one condition.
        """
        return hashlib.sha256(
            json.dumps(self.to_dict(), sort_keys=True,
                       separators=(",", ":")).encode("utf-8")).hexdigest()

    def readback_expectations(self) -> tuple[tuple[str, str], ...]:
        """`(status field, expected value)` pairs this recipe's env must produce at launch."""
        env = self.env or {}
        out: list[tuple[str, str]] = []
        for check in self.env_readback:
            try:
                field_name, expect = check["field"], check["expect"]
            except (TypeError, KeyError) as exc:
                raise RecipeError(
                    f"env_readback entry {check!r} needs 'field' and 'expect'") from exc
            if isinstance(expect, str):
                out.append((field_name, expect))
                continue
            var = check.get("env")
            if not var:
                raise RecipeError(
                    f"env_readback for {field_name!r} maps expectations by env value but "
                    f"names no 'env' variable to read them from")
            state = env.get(var, UNSET)
            if state not in expect:
                raise RecipeError(
                    f"env_readback for {field_name!r} does not cover {var}={state!r} "
                    f"(declared: {sorted(expect)}). Declare BOTH directions or drop the "
                    f"check -- an undeclared state is an UNCHECKED arm, and a control that "
                    f"is secretly the treatment is unrecoverable after the fact.")
            out.append((field_name, str(expect[state])))
        return tuple(out)

    def server_env(self, build_dir: Path, *,
                   base: Mapping[str, str] | None = None) -> dict[str, str]:
        """The exact environment the server is launched with: inherited env, then the
        loader pin, then the recipe's own `env` LAST.

        Precedence is recipe > loader > inherited, and it is total only because the loader's
        own variables are refused to `env` outright (`LOADER_OWNED_ENV`) -- so the recipe
        can override any host variable it likes without ever being able to silently drop
        the linkage pin the residency check depends on.
        """
        env = dict(os.environ if base is None else base)
        env["LD_LIBRARY_PATH"] = str(Path(build_dir) / "bin")
        env.update(self.env or {})
        return env

    def with_env(self, *, name: str | None = None, **overrides: str | None) -> "Recipe":
        """A sibling recipe differing only in `env` -- the two arms of a paired env A/B from
        ONE recipe file, with no JSON editing. A `None` value REMOVES a variable, so the
        control arm of an ON recipe is `with_env(KNOB=None)`.

        The name gets the override appended by default, because the serving floor is keyed
        by recipe NAME (`serving-floor.<name>.json`) and a different env is a different
        measured condition: reusing the name would silently judge the new arm against the
        old arm's floor. Pass `name=` to override deliberately.
        """
        merged = dict(self.env or {})
        for key, value in overrides.items():
            if value is None:
                merged.pop(key, None)
            else:
                merged[key] = value
        if name is None:
            name = self.name + "".join(
                f"+{k}={UNSET if v is None else v}" for k, v in sorted(overrides.items()))
        return replace(self, name=name, env=merged)

    def server_argv(self, build_dir: Path, port: int) -> list[str]:
        argv = ["taskset", "-c", self.cpu_list] if self.cpu_list else []
        argv += [str(Path(build_dir) / "bin" / "llama-server"),
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
        """One line naming every condition the number is conditional on.

        `env` is rendered even when empty (`env=none`), like `cpu=unpinned`: a record must
        never be able to claim an arm it did not run, and the reader of a serving row can
        only tell the treatment from the control by what the recipe says it launched.
        """
        sd = self.spec_decode.get("type", "none")
        pin = f" cpu={self.cpu_list}" if self.cpu_list else " cpu=unpinned"
        env = self.env or {}
        envs = (" env=" + ",".join(f"{k}={v}" for k, v in sorted(env.items()))
                if env else " env=none")
        rb = self.readback_expectations()
        rbs = " readback=" + ",".join(f"{f}={v}" for f, v in rb) if rb else ""
        return (f"{self.name} [np{self.np} {sd} {self.metric}{pin}{envs}{rbs} "
                f"#{self.recipe_hash[:12]}]")


class RecipeError(ValueError):
    """A recipe declares something that cannot be measured honestly."""


class EnvReadbackFailed(RuntimeError):
    """The launched server does not show the state its env claims to have set.

    "I set the knob" is not evidence the knob took effect. This is raised in BOTH
    directions -- a CONTROL arm that is secretly the treatment is unrecoverable after
    the fact, so verifying only that ON is on is not verification.
    """


class ServerDied(RuntimeError):
    """The server exited during load or measurement -- a build/config fault, not noise."""


def _status_fields(text: str) -> dict[str, str]:
    """`/proc/<pid>/status` as a mapping. `THP_enabled:\t1` -> {"THP_enabled": "1"}."""
    out: dict[str, str] = {}
    for line in text.splitlines():
        key, sep, value = line.partition(":")
        if sep:
            out[key.strip()] = value.strip()
    return out


def verify_env_readback(recipe: Recipe, pid: int, *,
                        status_text: str | None = None) -> dict[str, str | None]:
    """Prove the recipe's env arm took effect on the LAUNCHED process, or refuse.

    Fail-closed in both directions and in every failure mode: a mismatch raises, a field the
    kernel does not expose raises, and an unreadable `/proc` raises. It aborts rather than
    warns because the failure this guards is a mislabelled CONTROL arm -- if the control
    silently ran as the treatment, the A/B measured nothing and nothing downstream can tell.

    `pid` is `Popen.pid`, which is the SERVER even under `taskset`: taskset `exec`s the
    binary in place rather than forking, and `PR_SET_THP_DISABLE` survives `exec`.
    """
    expectations = recipe.readback_expectations()
    if not expectations:
        return {}
    if status_text is None:
        try:
            status_text = Path(f"/proc/{pid}/status").read_text()
        except OSError as exc:
            raise EnvReadbackFailed(
                f"{recipe.describe()}: cannot read /proc/{pid}/status ({exc}), so the env "
                f"arm is unverified -- refusing the measurement.") from exc
    fields = _status_fields(status_text)
    observed: dict[str, str | None] = {}
    for field_name, expect in expectations:
        got = fields.get(field_name)
        observed[field_name] = got
        if got != expect:
            raise EnvReadbackFailed(
                f"{recipe.describe()}: /proc/{pid}/status {field_name}={got!r}, but this "
                f"recipe's env requires {expect!r}. Setting the variable is not evidence it "
                f"took effect -- refusing the measurement.")
    return observed


def _measure_once(recipe: Recipe, build_dir: Path, port: int,
                  boot_timeout_s: int = 360) -> float:
    """Launch the server under `recipe`, fire `np` concurrent requests, return aggregate
    tok/s. The server is always stopped, even on error."""
    argv = recipe.server_argv(build_dir, port)
    srv = subprocess.Popen(argv, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                           env=recipe.server_env(build_dir))
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
        # The env arm is verified on the LIVE process, before a single token is measured.
        verify_env_readback(recipe, srv.pid)

        def one(i: int) -> tuple[int, float]:
            body = json.dumps({"prompt": _PROMPTS[i % len(_PROMPTS)],
                               "n_predict": recipe.n_predict, "temperature": recipe.temperature,
                               "top_p": recipe.top_p, "top_k": recipe.top_k,
                               "cache_prompt": False}).encode()
            req = urllib.request.Request(f"http://127.0.0.1:{port}/completion", data=body,
                                         headers={"Content-Type": "application/json"})
            t = json.loads(urllib.request.urlopen(req, timeout=600).read()).get("timings", {})
            # per-request decode rate, NOT wall-clock: each slot reports its own
            # predicted_n / predicted_ms, so the aggregate is the sum of the concurrent
            # slots' rates and is immune to the scheduling-tail jitter that made the
            # wall-clock aggregate ~5-10% noisy even at greedy (R23-43).
            return int(t.get("predicted_n", 0)), float(t.get("predicted_per_second", 0.0))

        # Warmup: one full np-wide round discarded, so cold-cache/clock-ramp does not
        # land in the measured sample (the first calibration run read high, then settled).
        with cf.ThreadPoolExecutor(recipe.np) as ex:
            list(ex.map(one, range(recipe.np)))
        with cf.ThreadPoolExecutor(recipe.np) as ex:
            rows = list(ex.map(one, range(recipe.np)))
        toks = [n for n, _ in rows]
        if min(toks) < recipe.n_predict // 2:
            raise ServerDied(f"degenerate measurement: tokens={toks}")
        return sum(rate for _, rate in rows)
    finally:
        srv.terminate()
        try:
            srv.wait(30)
        except Exception:
            srv.kill()
            srv.wait(10)


def _spread(runs: Sequence[float]) -> dict:
    """Per-arm dispersion, in the SAME grammar the serving floor is defined in.

    WHY A MEANS-ONLY COMPARATOR IS NOT ENOUGH. The `GGML_NOHUGEPAGE_PROCESS` effect on the
    CPU surface was a COMPRESSED DOWNSIDE TAIL, not a shifted mean: between-launch sd fell
    25.3x (2.510% -> 0.481%) while the median moved +5.23%. A comparator that reports only
    medians can return "no effect" on a real one -- the arm that removes the bad launches
    looks identical to the arm that keeps them once the tail is averaged away. So every
    serving row now carries, per arm, the same p95-deviation-from-median the floor itself is
    (`calibrate_floor`), plus sd/min/max and the per-launch values.

    REPORTING ONLY. No decision rule reads these fields; nothing here changes a verdict.
    """
    runs = list(runs)
    med = statistics.median(runs)
    devs = sorted(abs(r / med - 1.0) * 100.0 for r in runs) if med else [0.0] * len(runs)
    p95 = devs[min(len(devs) - 1, int(round(0.95 * (len(devs) - 1))))]
    sd = statistics.pstdev(runs)
    return {"n": len(runs), "median": med, "mean": statistics.fmean(runs), "sd": sd,
            "cv_pct": round(sd / med * 100.0, 3) if med else None,
            "min": min(runs), "max": max(runs),
            "range_pct": round((max(runs) - min(runs)) / med * 100.0, 3) if med else None,
            # p95 |deviation from median|, IDENTICAL in definition to `floor_pct`, so an
            # arm's own spread is directly comparable to the floor it must clear.
            "p95_dev_pct": round(p95, 3), "max_dev_pct": round(devs[-1], 3),
            "runs": runs}


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
            # The recipe is part of the artifact, so its identity travels with the number:
            # a reader (or a gate) can tell whether this row and the floor it was judged
            # against were even produced under the same launch conditions (R23-59).
            "recipe_hash": recipe.recipe_hash, "recipe_env": dict(recipe.env or {}),
            "recipe_describe": recipe.describe(),
            "metric": recipe.metric, "np": recipe.np, "pairs": pairs,
            "anchor_tok_s": a_med, "candidate_tok_s": c_med,
            "effect": effect, "effect_pct": effect * 100.0,
            "noise_floor_pct": floor_pct, "decisive": decisive,
            "anchor_samples": a_runs, "candidate_samples": c_runs,
            # Reporting only -- no decision rule reads these (see `_spread`).
            "anchor_spread": _spread(a_runs), "candidate_spread": _spread(c_runs)}


def calibrate_floor(recipe: Recipe, build_dir: Path, *, samples: int, port: int = 18311) -> dict:
    """A/A the serving metric `samples` times on ONE build: the run-to-run spread IS the
    noise floor a keep must clear. floor = p95 of |pairwise effect| against the median,
    reported at a few sample counts so a keep at N pairs is judged against the N-pair bar."""
    runs = [_measure_once(recipe, build_dir, port) for _ in range(samples)]
    # `floor_pct` IS this arm's p95 deviation from its own median -- taken from `_spread`
    # so the floor and the per-arm spread reported by `compare` can never drift apart.
    sp = _spread(runs)
    return {"schema": "epyc.autokernel.serving_floor.v1", "recipe": recipe.name,
            "recipe_hash": recipe.recipe_hash, "recipe_env": dict(recipe.env or {}),
            "recipe_describe": recipe.describe(),
            "metric": recipe.metric, "np": recipe.np, "samples": samples,
            "median_tok_s": sp["median"], "floor_pct": sp["p95_dev_pct"],
            "runs": runs, "cv_pct": sp["cv_pct"], "spread": sp}


__all__ = ["LOADER_OWNED_ENV", "RECIPE_SCHEMA", "UNSET", "EnvReadbackFailed", "Recipe",
           "RecipeError", "ServerDied", "calibrate_floor", "compare", "verify_env_readback"]
