"""AKX-P2d: floor job for the production workloads whose recipe is TWO pinned NUMA-half instances (W3, W4, W6).

The single-server calibration job (`cpu_aa_calibration`) launches one server. W3 (Qwen3.6-35B-A3B-MTP Q8_0, frontdoor
:8080/:8180), W4 and W6 serve from two instances, one per socket half, decoding concurrently. This module is the
instrument extension: recipe table, per-instance pinned argv, a balanced seeded A/A schedule, a concurrent two-instance
visit driver (injectable server class, so it is testable without a model) and a record builder that hands the per-surface
samples to `cpu_aa_calibration.build_record` (unit=process, n >= 24 pairs, quiet/co-tenant label).

A visit = both instances started pinned, both decode the same prompt CONCURRENTLY (the production condition), both stopped.
Samples per visit: one per instance plus the aggregate (sum of 1000/step_ms, tok/s-equivalent). A/A: both arms are the same
store/env; the paired scatter is the floor. A visit is valid only if both instances produced the full token count.

Recipes are data. A recipe whose model file is absent is reported by `missing_inputs` (FUTILITY_MISSING_INPUT in the row job),
never silently substituted: W4 (gemma-4-26B-A4B) and W6 (Qwen3-Next-80B-A3B) were retired from the registry 2026-09-22 and their
GGUFs are no longer on disk; their entries stay so the same job runs them if a model is restored.
"""
from __future__ import annotations

import concurrent.futures as cf
import hashlib
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from . import cpu_aa_calibration as aa

HALF_A = "0-47,96-143"
HALF_B = "48-95,144-191"
CHAT_TEMPLATE_W3 = ("/mnt/raid0/llm/models/chat-templates/sha256/"
                    "1443ea9ab4bb0a2663b1002a5d4baad970e9f0b2f42ca4ec12b4683838284551.jinja")


class TwoInstanceError(ValueError):
    """Recipe or schedule is malformed, or a visit cannot be a valid sample."""


@dataclass(frozen=True)
class Instance:
    cpus: str          # taskset list, e.g. "48-95,144-191"
    numa_nodes: str    # numactl --interleave=<nodes>
    port: int


@dataclass(frozen=True)
class Recipe:
    name: str
    workload: str
    model: str
    instances: tuple[Instance, ...]
    threads: int
    server_args: tuple[str, ...]
    env: Mapping[str, str] = field(default_factory=dict)
    retired: str = ""   # non-empty: why the production workload no longer exists

    def digest(self) -> str:
        blob = json.dumps({"n": self.name, "m": self.model, "i": [vars(i) for i in self.instances], "t": self.threads,
                           "a": list(self.server_args), "e": dict(sorted(self.env.items()))}, sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()


_ENV = {"GGML_FA_SPLIT_KV": "0", "GGML_FUSED_DECODE_OFF": "1", "GGML_IQK": "1", "GGML_NOHUGEPAGE_PROCESS": "1",
        "OMP_DYNAMIC": "false", "OMP_PLACES": "cores", "OMP_PROC_BIND": "spread", "OMP_WAIT_POLICY": "active"}

W3 = Recipe(
    name="W3-qwen36-35b-a3b-mtp-q8-2inst", workload="W3",
    model="/mnt/raid0/llm/models/Qwen3.6-35B-A3B-MTP-Q8_0.gguf",
    instances=(Instance(HALF_A, "0,1", 8080), Instance(HALF_B, "2,3", 8180)), threads=48,
    server_args=("-np", "1", "-c", "262144", "-ub", "2048", "--flash-attn", "on", "--jinja", "-ctk", "q8_0", "-ctv", "q8_0",
                 "--no-mmap", "--chat-template-file", CHAT_TEMPLATE_W3, "--spec-type", "draft-mtp",
                 "--spec-draft-n-max", "4", "--reasoning", "off", "--device", "none", "--device-draft", "none",
                 "--no-kv-unified"),
    env=_ENV)
W4 = Recipe(
    name="W4-gemma4-26b-a4b-mtp-q4km-2inst", workload="W4",
    model="/mnt/raid0/llm/models/gemma-4-26B-A4B-it-ORIG-Q4_K_M.gguf",
    instances=(Instance(HALF_A, "0,1", 8082), Instance(HALF_B, "2,3", 8182)), threads=48,
    server_args=("-np", "1", "-c", "32768", "-ub", "512", "--flash-attn", "on", "--jinja", "--no-mmap",
                 "--spec-type", "draft-mtp", "--spec-draft-n-max", "2", "--device", "none", "--device-draft", "none",
                 "--no-kv-unified"),
    env=_ENV, retired="registry retired the gemma4-26B-A4B workers 2026-09-22 (tenancy moved to W3 frontdoor)")
W6 = Recipe(
    name="W6-qwen3-next-80b-a3b-q4km-2inst", workload="W6",
    model="/mnt/raid0/llm/models/Qwen3-Next-80B-A3B-Instruct-Q4_K_M.gguf",
    instances=(Instance(HALF_A, "0,1", 8085), Instance(HALF_B, "2,3", 8185)), threads=48,
    server_args=("-np", "1", "-c", "131072", "--flash-attn", "on", "--jinja", "--no-mmap", "--spec-type", "none",
                 "--device", "none", "--no-kv-unified"),
    env=_ENV, retired="registry retired Qwen3-Next-80B-A3B (ingest_long_context) 2026-09-22 (now the :8083 27B GPU alias)")
RECIPES = {r.workload: r for r in (W3, W4, W6)}


def instance_argv(recipe: Recipe, store: str, idx: int, *, port: int | None = None, host: str = "127.0.0.1") -> list[str]:
    """numactl --interleave=<half nodes> -- taskset -c <half> llama-server ...; literal thread count, one half per instance."""
    inst = recipe.instances[idx]
    return ["/usr/bin/taskset", "-c", inst.cpus, "/usr/bin/numactl", f"--interleave={inst.numa_nodes}",
            f"{store}/bin/llama-server", "-m", recipe.model, "--host", host, "--port", str(port or inst.port),
            "-t", str(recipe.threads), "-tb", str(recipe.threads), *recipe.server_args]


def validate_recipe(recipe: Recipe) -> None:
    if len(recipe.instances) != 2:
        raise TwoInstanceError("a two-instance recipe has exactly two instances")
    a, b = (set(_expand(i.cpus)) for i in recipe.instances)
    if a & b:
        raise TwoInstanceError("instance cpu sets overlap; halves must be disjoint")
    if recipe.instances[0].port == recipe.instances[1].port:
        raise TwoInstanceError("instances share a port")
    for i in recipe.instances:
        if len(_expand(i.cpus)) < recipe.threads:
            raise TwoInstanceError(f"{i.cpus} has fewer cpus than threads={recipe.threads}")


def _expand(s: str) -> list[int]:
    out: list[int] = []
    for part in s.split(","):
        if "-" in part:
            lo, hi = part.split("-")
            out += range(int(lo), int(hi) + 1)
        elif part:
            out.append(int(part))
    return out


def missing_inputs(recipe: Recipe, store: str) -> list[str]:
    need = [recipe.model, f"{store}/bin/llama-server"]
    return [p for p in need if not Path(p).is_file() or Path(p).stat().st_size == 0]


def schedule(seed: int, pairs: int) -> list[tuple[int, int, str]]:
    """2*pairs (visit, pair, arm); half the pairs AB, half BA, seeded. Same shape as the single-server job."""
    if pairs % 2:
        raise TwoInstanceError("pairs must be even")
    orders = ["AB"] * (pairs // 2) + ["BA"] * (pairs // 2)
    random.Random(seed).shuffle(orders)
    return [(2 * p + j, p, arm) for p, o in enumerate(orders) for j, arm in enumerate(o)]


def step_ms(r: Mapping[str, Any]) -> float:
    return r["predicted_ms"] / (r["predicted_n"] - (r.get("draft_n_accepted") or 0))


def run_visit(servers: Sequence[Any], prompt: str, n_predict: int, *, expected_tokens: int | None = None) -> dict:
    """servers: two objects with start()/complete(text, n)/stop(). Both start, then decode CONCURRENTLY, then both stop.

    Returns {"instances": [{step_ms, tok_s, predicted_n}, ...], "aggregate_tok_s", "valid"}; a failure on either
    instance makes the visit invalid (never a one-sided sample)."""
    if len(servers) != 2:
        raise TwoInstanceError("two servers required")
    started: list[Any] = []
    try:
        for s in servers:
            s.start()
            started.append(s)
        with cf.ThreadPoolExecutor(max_workers=2) as ex:
            futs = [ex.submit(s.complete, prompt, n_predict) for s in servers]
            res = [f.result() for f in futs]
    finally:
        for s in started:
            s.stop()
    want = expected_tokens or n_predict
    inst = []
    for r in res:
        ok = r.get("predicted_n") is not None and r["predicted_n"] >= want
        inst.append({"valid": bool(ok), "predicted_n": r.get("predicted_n"),
                     "step_ms": step_ms(r) if ok else None, "tok_s": r.get("tok_s")})
    valid = all(i["valid"] for i in inst)
    agg = sum(1000.0 / i["step_ms"] for i in inst) if valid else None
    return {"instances": inst, "aggregate_tok_s": agg, "valid": valid}


def surfaces_from_visits(rows: Sequence[Mapping]) -> dict[str, dict]:
    """rows: {pair, arm, valid, instances, aggregate_tok_s}. Per-surface paired samples (only pairs valid in BOTH arms).
    Surfaces: agg (gating; higher-is-better tok/s) and i0/i1 (informational, step ms)."""
    by: dict[int, dict[str, Mapping]] = {}
    for r in rows:
        by.setdefault(r["pair"], {})[r["arm"]] = r
    surf: dict[str, dict] = {"AGG": {"anchor": [], "candidate": [], "gating": True},
                             "I0": {"anchor": [], "candidate": [], "gating": False},
                             "I1": {"anchor": [], "candidate": [], "gating": False}}
    for p in sorted(by):
        x, y = by[p].get("A"), by[p].get("B")
        if not (x and y and x["valid"] and y["valid"]):
            continue
        surf["AGG"]["anchor"].append(x["aggregate_tok_s"])
        surf["AGG"]["candidate"].append(y["aggregate_tok_s"])
        for k in (0, 1):
            surf[f"I{k}"]["anchor"].append(x["instances"][k]["step_ms"])
            surf[f"I{k}"]["candidate"].append(y["instances"][k]["step_ms"])
    return surf


def build_floor_record(recipe: Recipe, store_digest: str, rows: Sequence[Mapping], *, contention_label: str,
                       foreign_busy_p95: Sequence[float], source_row: str, serving_pairs: int = 4,
                       numa: str | None = None) -> dict:
    validate_recipe(recipe)
    surf = surfaces_from_visits(rows)
    host_state = {"store_digest": store_digest, "cpu_list": ";".join(i.cpus for i in recipe.instances),
                  "threads": recipe.threads, "numa": numa or ";".join(f"interleave={i.numa_nodes}" for i in recipe.instances),
                  "recipe_env_digest": hashlib.sha256(json.dumps(dict(sorted(recipe.env.items()))).encode()).hexdigest()[:16]}
    surf = {k: v for k, v in surf.items() if v["anchor"]}
    rec = aa.build_record(recipe=recipe.name, recipe_hash=recipe.digest(), store_digest=store_digest, unit=aa.UNIT_PROCESS,
                          host_state=host_state, contention_label=contention_label, foreign_busy_p95=foreign_busy_p95,
                          surfaces=surf, serving_pairs=serving_pairs, source_row=source_row)
    rec["topology"] = "two-instance"
    rec["workload"] = recipe.workload
    rec["unit_of_sample"] = "one visit = both instances launched, decoded concurrently, stopped"
    return rec
