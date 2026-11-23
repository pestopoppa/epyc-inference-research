#!/usr/bin/env python3
"""Offline instrument: re-score a dice.v1 acceptance capture at other sampling temperatures.

    python3 -m autokernel.loop.temperature_rescore --capture CAP --units RESULTS_JSONL \\
        [--temps 0 0.2 1.0] [--out rescore.json]

Promoted from the DICE replay (research 03c100462, scripts/kernel_rnd/dice_replay/dice_replay.py)
so the loop can ask "what does this acceptance capture say at the served temperature?" instead of
trusting greedy-only evidence. No server, no GPU, no live run: it reads the capture.

How a temperature is re-applied. The capture stores each verified row's target top-64 logits and
the served sampler's post-sampler candidates. llama.cpp's default chain filters (top_k, top_p, min_p)
on the T=1 distribution and applies temperature LAST, so the candidate set does not move with T
and only the probabilities do: p_T = softmax(logit / T) over the survivors. T = 0 is a point mass
on the argmax. `check_sampler` re-derives the capture's own post-sampler lists at the capture T and
reports the largest deviation, which is the proof this model of the sampler is the served one.

What it reports. Per (T, W): V0 (the served sample-and-compare chain, exact expectation) and V2
(draw-blind Gumbel-walk tree at node budget W, exact expectation) in tokens per target pass. The
Monte-Carlo arms (V1, V3) stay in dice_replay.py. Counterfactual rounds are replayed on the served
prefixes of the capture's `source_T` trajectories (the DICE approximation), so every output arm is
a PROJECTION (`kind: "projection"`), never a measured arm.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

from . import sampling_axis

SCHEMA = "epyc.autokernel.temperature_rescore.v1"
DEFAULT_TEMPS = (0.0, 0.2, 1.0)
HOLDOUT_PROMPTS = range(32, 64)
DICE_REPLAY = Path(__file__).resolve().parents[2] / "dice_replay" / "dice_replay.py"


def load_dice(path: Path = DICE_REPLAY):
    """The replay primitives (norm, blind_shape, v0/v2 analytics, WS), loaded by path + digest."""
    spec = importlib.util.spec_from_file_location("dice_replay_core", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _softmax(pairs: list[tuple[int, float]], temperature: float) -> dict[int, float]:
    top = max(v for _, v in pairs)
    weights = [(i, math.exp((v - top) / temperature)) for i, v in pairs]
    total = sum(w for _, w in weights)
    return {i: w / total for i, w in weights}


def sampler_probs(logits: Iterable, temperature: float, *, top_k: int, top_p: float,
                  min_p: float) -> dict[int, float]:
    """Post-sampler candidate distribution at `temperature` (filters on T=1, temperature last)."""
    pairs = sorted(((int(i), float(v)) for i, v in logits), key=lambda x: -x[1])
    if temperature == 0:
        return {pairs[0][0]: 1.0}
    base = sorted(_softmax(pairs, 1.0).items(), key=lambda x: -x[1])[:top_k]
    norm = sum(p for _, p in base)
    kept, cum = [], 0.0
    for i, p in base:
        kept.append((i, p / norm))
        cum += p / norm
        if cum >= top_p:
            break
    norm = sum(p for _, p in kept)
    ceiling = max(p for _, p in kept) / norm
    ids = {i for i, p in kept if p / norm >= min_p * ceiling}
    return _softmax([(i, v) for i, v in pairs if i in ids], temperature)


def _rounds(capture: Path, unit: Mapping[str, Any]) -> Iterable[dict]:
    with open(capture, "rb") as handle:
        handle.seek(unit["off0"])
        blob = handle.read(unit["off1"] - unit["off0"])
    for line in blob.decode().splitlines():
        rec = json.loads(line)
        if rec.get("k") == "round":
            yield rec


def _rows(rec: Mapping[str, Any]) -> int:
    n = 0
    for row in rec["rows"]:
        if row is None:
            break
        n += 1
    return n


def check_sampler(rec: Mapping[str, Any]) -> float:
    """Largest |p_rederived - p_captured| over the round's multi-candidate rows."""
    worst = 0.0
    for row in rec["rows"]:
        if row is None or len(row[1]) < 2:
            continue
        mine = sampler_probs(row[0], rec["T"], top_k=rec["top_k"], top_p=rec["top_p"], min_p=rec["min_p"])
        theirs = {int(i): float(p) for i, p in row[1]}
        worst = max(worst, *(abs(mine.get(i, 0.0) - theirs.get(i, 0.0)) for i in set(mine) | set(theirs)))
    return worst


def rescore_round(rec: Mapping[str, Any], temperature: float, dice, ws: Iterable[int]) -> dict[str, Any]:
    # At the capture's own temperature the captured post-sampler lists ARE the served
    # distribution (this is the exact replay); elsewhere they are re-derived from the logits.
    """Exact V0 / V2 accepted-draft expectations for one captured round at `temperature`."""
    L = max(0, min(rec["D"], len(rec["chain"]), len(rec["mtp_top"]), _rows(rec) - 1))
    chain = rec["chain"][:L]
    if temperature == rec["T"]:
        ps = [dice.norm({int(c): float(p) for c, p in rec["rows"][j][1]}) for j in range(L)]
    else:
        ps = [dice.norm(sampler_probs(rec["rows"][j][0], temperature, top_k=rec["top_k"],
                                      top_p=rec["top_p"], min_p=rec["min_p"])) for j in range(L)]
    qs = [dice.norm({int(t[0]): float(t[2]) for t in rec["mtp_top"][j] if float(t[2]) > 0}) for j in range(L)]
    for j in range(L):  # same drafter-q conventions as dice_replay.replay_round
        if not qs[j]:
            qs[j] = {chain[j]: 1.0}
        if chain[j] not in qs[j]:
            qs[j][chain[j]] = min(qs[j].values()) * 1e-3
            qs[j] = dice.norm(qs[j])
    out = {}
    for W in ws:
        shape = dice.blind_shape(chain, qs, W)
        out[W] = {"V0": dice.v0_analytic(chain, ps, W), "V2": dice.v2_analytic(shape, chain, ps)}
    return out


def source_temperature(target: float, available: Iterable[float]) -> float:
    """Capture temperature whose trajectories are re-scored: exact match, else the nearest (lower on ties)."""
    pool = sorted(set(available))
    return min(pool, key=lambda t: (round(abs(t - target), 9), t))


def rescore(capture: Path, units: list[Mapping[str, Any]], temps: Iterable[float] = DEFAULT_TEMPS, *,
            prompts: Iterable[int] | None = None, ws: Iterable[int] | None = None,
            dice=None, source_for: Mapping[float, float] | None = None) -> dict[str, Any]:
    """Re-score `capture` at each temperature; returns the instrument record (see module doc)."""
    dice_module, dice_sha = (dice, None) if dice is not None else load_dice()
    ws = tuple(ws or dice_module.WS)
    wanted = None if prompts is None else set(prompts)
    available = {float(u["T"]) for u in units}
    sampler = dict(next(iter(units)).get("check") or {})
    top_k, top_p = sampler.get("top_k"), sampler.get("top_p")
    rows_out, deviation, arms = [], 0.0, []
    for target in (float(t) for t in temps):
        src = float((source_for or {}).get(target, source_temperature(target, available)))
        totals = {W: {"V0": 0.0, "V2": 0.0} for W in ws}
        n = 0
        for unit in units:
            if float(unit["T"]) != src or (wanted is not None and unit["prompt_idx"] not in wanted):
                continue
            for rec in _rounds(capture, unit):
                if target == src:
                    deviation = max(deviation, check_sampler(rec))
                per_w = rescore_round(rec, target, dice_module, ws)
                for W in ws:
                    for key in ("V0", "V2"):
                        totals[W][key] += per_w[W][key]
                n += 1
        for W in ws:
            rows_out.append({"T": target, "source_T": src, "W": W, "rounds": n,
                             **{k: (totals[W][k] / n + 1 if n else None) for k in ("V0", "V2")}})
        arms.append({"temperature": target, "top_p": top_p, "top_k": top_k,
                     "exact_path": None, "kind": "projection",
                     "basis": f"offline rescore of capture trajectories at source T={src:g}"})
    return {"schema": SCHEMA, "unit": "expected tokens per target pass (accepted drafts + 1)",
            "capture": {"name": Path(capture).name, "sha256": hashlib.sha256(Path(capture).read_bytes()).hexdigest()},
            "dice_replay_sha256": dice_sha, "sampler": {"top_k": top_k, "top_p": top_p, "min_p": sampler.get("min_p")},
            "sampler_check_max_deviation_at_capture_T": deviation,
            "sampling": sampling_axis.normalize(arms), "rows": rows_out}


def load_units(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--capture", type=Path, required=True)
    ap.add_argument("--units", type=Path, required=True, help="results.jsonl unit index (off0/off1 per unit)")
    ap.add_argument("--temps", type=float, nargs="+", default=list(DEFAULT_TEMPS))
    ap.add_argument("--holdout", action="store_true", help="prompts 32-63 only (the DICE holdout)")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args(argv)
    record = rescore(args.capture, load_units(args.units), args.temps,
                     prompts=HOLDOUT_PROMPTS if args.holdout else None)
    text = json.dumps(record, indent=1, sort_keys=True) + "\n"
    (args.out.write_text(text, encoding="utf-8") if args.out else sys.stdout.write(text))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
