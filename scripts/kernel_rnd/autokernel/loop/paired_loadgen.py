"""AKR-SG-2: paired A/B/A/B alternation wrapper around a load generator.

The reference load generators (inference-perf et al.) report single-run percentiles: no
pairing, no floor. The serving gate is a paired, alternated A/B judged against a measured
noise floor, so this wraps ANY generator callable in that discipline and emits the existing
serving-gate record (`epyc.autokernel.serving_ab.v1`, same keys `serving.compare` writes).

DECISION (taken at the start, as the task requires): OWN CLIENT, not a wrapper over
kubernetes-sigs/inference-perf. inference-perf has no llama-server adapter (server.type is
vllm/sglang/tgi/mock), infers its tokenizer from an HF model name, scrapes vLLM-only
metrics, and pins python>=3.12 + transformers==5.13.0; the adapter alone is 0.5-1 day and
leaves the paired statistic still to build. The client here is the same closed-round shape
as `serving._measure_once` (np concurrent /completion requests, aggregate = sum of each
slot's reported `predicted_per_second`) plus the AKR-SG-1 scheduling self-check, so a
candidate generator can still be dropped in as an `Arm.measure` callable later. KPF-64's
adapter need is unaffected (it is the adapter, not this wrapper).

Design points:
  * order is A,B,A,B,... per pair, FIXED (no coin flip), and every pair's two measurements
    share `seed = base_seed + pair_index`, so both arms see the identical request plan.
  * effect = median(B)/median(A) - 1 (same estimator as `serving.compare`); per-pair deltas
    are kept and reported.
  * floor: p95 |pair delta| of an A/A run of `aa_pairs` pairs through the same alternation
    (the `session` unit: readings are separate request rounds against one running server).
    A caller may instead pass a previously measured `floor_pct` with its unit.
  * a sample whose self-check voided is never averaged: the run raises `LoadDriverVoided`.
Runs against real servers need a window; the toy-endpoint test needs none.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
import concurrent.futures as cf
import hashlib
import json
import statistics
import time
import urllib.request

from . import loadgen_selfcheck as lgs
from . import serving

SCHEMA = "epyc.autokernel.serving_ab.v1"
GENERATOR = "epyc.autokernel.paired_loadgen.v1"


@dataclass
class Arm:
    name: str
    #: measure(seed) -> (aggregate_rate, selfcheck_report_or_None)
    measure: Callable[[int], tuple[float, dict | None]]


def http_arm(name: str, base_url: str, *, np: int, n_predict: int = 16,
             timeout_s: float = 60.0) -> Arm:
    """Closed round of `np` concurrent /completion requests; value = sum of reported
    predicted_per_second (the serving gate's aggregate). Start lag is self-checked."""
    def measure(seed: int):
        log = lgs.SchedulingLog()
        release = time.monotonic()

        def one(i: int):
            log.record(release, time.monotonic())
            body = json.dumps({"prompt": f"seed {seed} slot {i}", "n_predict": n_predict,
                               "seed": seed, "cache_prompt": False}).encode()
            req = urllib.request.Request(base_url + "/completion", data=body,
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=timeout_s) as r:
                return float(json.loads(r.read())["timings"]["predicted_per_second"])

        with cf.ThreadPoolExecutor(np) as ex:
            rates = list(ex.map(one, range(np)))
        return sum(rates), log.require_valid(requested_n=np)
    return Arm(name, measure)


def _effect(a: float, b: float) -> float:
    return b / a - 1.0


def run_pairs(a: Arm, b: Arm, pairs: int, base_seed: int):
    a_vals, b_vals, checks = [], [], []
    for k in range(pairs):
        seed = base_seed + k
        for arm, sink in ((a, a_vals), (b, b_vals)):      # A then B, every pair
            value, check = arm.measure(seed)
            sink.append(value)
            checks.append({"pair": k, "arm": arm.name, "seed": seed, "selfcheck": check})
    return a_vals, b_vals, checks


def paired_record(a: Arm, b: Arm, *, pairs: int, base_seed: int, metric: str = "aggregate_tok_s",
                  aa_pairs: int = 0, floor_pct: float | None = None,
                  floor_unit: str | None = None, name: str = "paired-loadgen") -> dict:
    if pairs < 2:
        raise ValueError("a paired run needs >= 2 pairs")
    floor_source = "supplied" if floor_pct is not None else None
    aa_deltas: list[float] = []
    if floor_pct is None and aa_pairs:
        # A/A through the identical alternation, disjoint seeds from the A/B run.
        x, y, _ = run_pairs(a, a, aa_pairs, base_seed + 1_000_003)
        aa_deltas = [abs(_effect(p, q)) * 100.0 for p, q in zip(x, y)]
        srt = sorted(aa_deltas)
        floor_pct = srt[min(len(srt) - 1, int(round(0.95 * (len(srt) - 1))))]
        floor_unit, floor_source = serving.UNIT_SESSION, f"A/A p95 |pair delta|, {aa_pairs} pairs"
    a_vals, b_vals, checks = run_pairs(a, b, pairs, base_seed)
    deltas = [_effect(p, q) * 100.0 for p, q in zip(a_vals, b_vals)]
    a_med, b_med = statistics.median(a_vals), statistics.median(b_vals)
    effect = _effect(a_med, b_med)
    if floor_pct is not None and floor_unit is None:
        raise ValueError("a supplied floor must state its unit")
    decisive = None if floor_pct is None else (abs(effect) * 100.0 >= floor_pct)
    ident = hashlib.sha256(json.dumps([a.name, b.name, metric, base_seed],
                                      sort_keys=True).encode()).hexdigest()
    return {"schema": SCHEMA, "recipe": name, "recipe_hash": ident, "recipe_env": {},
            "recipe_describe": f"{a.name} vs {b.name} via {GENERATOR}",
            "metric": metric, "np": None, "pairs": pairs,
            "anchor_tok_s": a_med, "candidate_tok_s": b_med,
            "effect": effect, "effect_pct": effect * 100.0,
            "effect_unit": serving.UNIT_SESSION, "floor_unit": floor_unit,
            "noise_floor_pct": floor_pct, "decisive": decisive,
            "anchor_samples": a_vals, "candidate_samples": b_vals,
            "anchor_spread": serving._spread(a_vals), "candidate_spread": serving._spread(b_vals),
            "residency": serving._residency_fold([]),
            "anchor_residency": [], "candidate_residency": [],
            "generator": {"name": GENERATOR, "order": "A,B per pair, fixed",
                          "base_seed": base_seed, "pair_seeds": [base_seed + k for k in range(pairs)],
                          "pair_delta_pct": deltas, "floor_source": floor_source,
                          "aa_pair_abs_delta_pct": aa_deltas, "selfchecks": checks}}


def validate_record(rec: dict) -> list[str]:
    """Structural check against the keys `serving.compare` writes; returns problems."""
    need = ("schema", "recipe", "recipe_hash", "metric", "pairs", "anchor_tok_s", "candidate_tok_s",
            "effect", "effect_pct", "effect_unit", "floor_unit", "noise_floor_pct", "decisive",
            "anchor_samples", "candidate_samples", "anchor_spread", "candidate_spread", "residency")
    bad = [f"missing {k}" for k in need if k not in rec]
    if rec.get("schema") != SCHEMA:
        bad.append("schema")
    if rec.get("noise_floor_pct") is not None and rec.get("floor_unit") not in serving.FLOOR_UNITS:
        bad.append("floor without admissible unit")
    return bad
