#!/usr/bin/env python3
"""Analysis of the R23-64 CPU decode thread sweep. UNIT = LAUNCH (FLOOR-UNIT-1).

Reads the per-launch <runs>/<arm>_r<round>.rows.jsonl written by client.py and reports, per arm:
  * the per-LAUNCH token-weighted decode rate  sum(pred_n) / sum(pred_ms/1000)  over prompts with
    pred_n >= 16 (the floor the client and every INF-70 analysis use);
  * n launches, mean, BETWEEN-LAUNCH sd, CV and a 95% CI (Student t, df = n-1);
and, against the control arm (t48):
  * the effect (arm / control - 1) with a percentile bootstrap 95% CI resampling LAUNCHES;
  * the minimum detectable effect (MDE) at alpha 0.05 two-sided, power 0.80, from the observed
    pooled between-launch sd, and (if a floor is supplied) from the floor's sd.

THE UNIT RULE. `-t` is set at server launch, so the experimental unit is the launch and the only
admissible floor is a LAUNCH floor (shim-ON: sd 0.609%, n=6, qwen38_flash_next_recipe.FLOORS
['launch_shim_on']). The within-session ARM floor (0.501%) is ~13x tighter and is NOT interchangeable
with it; substituting it is the error that sized a THP A/B 1200-fold too small. FLOOR-UNIT-1: a gate
asked to compare an effect against a floor of a different unit REFUSES. It does not warn. A floor
with no unit is inadmissible. This tool raises UnitRefusal (CLI exit 2) in both cases.
(session == one process launch in the INF-70 vocabulary, and `process` is FLOOR-UNIT-1's own word
for it, so launch / process / session are the one admissible unit.)

SCREENS. A launch whose contention screens (screen.py) say DROPPED is excluded BEFORE any statistic;
a launch with no sampler records (NO-SAMPLES) or fewer rows than the prompt mix, or any HTTP error,
is excluded too, unless --allow-unscreened is given. Excluded launches are listed with the reason.

  usage: analyze_sweep.py --runs DIR [--control t48] [--floor-sd-pct 0.609 --floor-unit launch --floor-n 6]
                          [--allow-unscreened] [--expected-rows 24] [--bootstrap 10000] [--json OUT]
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import random
import re
import statistics as st
import sys
from dataclasses import dataclass, field
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import screen as screens  # noqa: E402

FLOOR_TOKENS = 16               # pred_n >= 16: below it the per-prompt rate is dominated by the first-token cost
EXPECTED_ROWS = 24
LAUNCH_UNITS = ("launch", "process", "session")
LABEL_RE = re.compile(r"^(?P<arm>t\d+m?)_r(?P<round>\d+)$")
ALPHA, POWER = 0.05, 0.80
SEED = 20261023


class UnitRefusal(Exception):
    """FLOOR-UNIT-1: a gate asked to compare against a floor of the wrong (or no) unit refuses."""


# ---------------------------------------------------------------------------
# Student t, from the regularised incomplete beta (stdlib only)
# ---------------------------------------------------------------------------
def _betacf(a: float, b: float, x: float) -> float:
    tiny, qab, qap, qam = 1e-300, a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 300):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-14:
            break
    return h


def _betainc(a: float, b: float, x: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    front = math.exp(lbeta + a * math.log(x) + b * math.log(1.0 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def t_cdf(t: float, df: float) -> float:
    p = 0.5 * _betainc(df / 2.0, 0.5, df / (df + t * t))
    return 1.0 - p if t > 0 else p


def t_ppf(p: float, df: float) -> float:
    if not 0.0 < p < 1.0:
        raise ValueError("p must be in (0, 1)")
    if p == 0.5:
        return 0.0
    if p < 0.5:
        return -t_ppf(1.0 - p, df)
    lo, hi = 0.0, 1.0
    while t_cdf(hi, df) < p:
        hi *= 2.0
        if hi > 1e9:
            raise ValueError("t_ppf did not bracket")
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if t_cdf(mid, df) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ---------------------------------------------------------------------------
# Per-launch estimator
# ---------------------------------------------------------------------------
def launch_rate(rows: list[dict]) -> tuple[Optional[float], int, int]:
    """(token-weighted decode t/s, n prompts counted, n tokens) over rows with pred_n >= 16."""
    d = [r for r in rows if (r.get("pred_n") or 0) >= FLOOR_TOKENS and r.get("pred_ms")]
    if not d:
        return None, 0, 0
    toks = sum(r["pred_n"] for r in d)
    secs = sum(r["pred_ms"] for r in d) / 1000.0
    return (toks / secs if secs > 0 else None), len(d), toks


@dataclass
class LaunchRec:
    label: str
    arm: str
    round_no: int
    tps: Optional[float]
    n_dec: int
    status: str                      # valid | incomplete | empty | DROPPED | NO-SAMPLES
    detail: dict = field(default_factory=dict)


def _own_threads(arm: str) -> int:
    return int(re.match(r"t(\d+)", arm).group(1))


def load_launches(runs_dir: str, expected_rows: int = EXPECTED_ROWS, allow_unscreened: bool = False) -> list[LaunchRec]:
    out = []
    for path in sorted(glob.glob(os.path.join(runs_dir, "*.rows.jsonl"))):
        lab = os.path.basename(path)[: -len(".rows.jsonl")]
        m = LABEL_RE.match(lab)
        if not m:
            continue
        arm, rnd = m.group("arm"), int(m.group("round"))
        rows = [json.loads(x) for x in open(path) if x.strip()]
        tps, n_dec, _tok = launch_rate(rows)
        errs = sum(1 for r in rows if r.get("verdict") == "HTTP-ERROR")
        if errs or len(rows) < expected_rows:
            out.append(LaunchRec(lab, arm, rnd, tps, n_dec, "incomplete", dict(rows=len(rows), http_errors=errs)))
            continue
        if tps is None:
            out.append(LaunchRec(lab, arm, rnd, None, 0, "empty", {}))
            continue
        verdict, detail = screens.screen_launch(runs_dir, lab, _own_threads(arm))
        if verdict == "DROPPED" or (verdict == "NO-SAMPLES" and not allow_unscreened):
            out.append(LaunchRec(lab, arm, rnd, tps, n_dec, verdict, detail))
            continue
        out.append(LaunchRec(lab, arm, rnd, tps, n_dec, "valid", dict(screen=verdict)))
    return out


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def summarize(vals: list[float]) -> dict:
    n = len(vals)
    mean = st.fmean(vals)
    sd = st.stdev(vals) if n > 1 else None
    ci = None
    if sd is not None:
        half = t_ppf(1.0 - ALPHA / 2.0, n - 1) * sd / math.sqrt(n)
        ci = (mean - half, mean + half)
    return dict(n=n, mean=mean, sd=sd, cv_pct=(100.0 * sd / mean if sd is not None else None),
                ci95=ci, ci95_halfwidth_pct=(100.0 * (ci[1] - ci[0]) / 2.0 / mean if ci else None),
                min=min(vals), max=max(vals), values=list(vals))


def bootstrap_effect_pct(arm: list[float], ctl: list[float], reps: int = 10000, seed: int = SEED) -> tuple[float, float]:
    """Percentile 95% CI of (mean(arm)/mean(ctl) - 1)*100, resampling LAUNCHES within each arm."""
    rng = random.Random(seed)
    eff = []
    for _ in range(reps):
        a = st.fmean(rng.choices(arm, k=len(arm)))
        c = st.fmean(rng.choices(ctl, k=len(ctl)))
        eff.append((a / c - 1.0) * 100.0)
    eff.sort()
    return eff[int(round(0.025 * (reps - 1)))], eff[int(round(0.975 * (reps - 1)))]


def mde_pct(sd: float, n1: int, n2: int, mean_ctl: float) -> Optional[float]:
    """Smallest true difference of means detectable with probability POWER at two-sided ALPHA, in % of
    the control mean, for known-ish between-launch sd `sd` (absolute t/s)."""
    df = n1 + n2 - 2
    if df < 1 or sd is None:
        return None
    k = t_ppf(1.0 - ALPHA / 2.0, df) + t_ppf(POWER, df)
    return 100.0 * k * sd * math.sqrt(1.0 / n1 + 1.0 / n2) / mean_ctl


def pooled_sd(a: dict, b: dict) -> Optional[float]:
    if a["sd"] is None or b["sd"] is None:
        return None
    df = a["n"] + b["n"] - 2
    return math.sqrt(((a["n"] - 1) * a["sd"] ** 2 + (b["n"] - 1) * b["sd"] ** 2) / df)


@dataclass(frozen=True)
class Floor:
    sd_pct: float
    unit: Optional[str]
    n: Optional[int] = None


def check_floor(floor: Optional[Floor]) -> None:
    """FLOOR-UNIT-1: refuse, do not warn."""
    if floor is None:
        return
    if not floor.unit:
        raise UnitRefusal("floor has no unit: a floor with no unit is inadmissible as a gate input (FLOOR-UNIT-1 rule 1)")
    if floor.unit.lower() not in LAUNCH_UNITS:
        raise UnitRefusal(f"floor unit {floor.unit!r} != 'launch': this sweep's unit is the LAUNCH (-t is a launch-time flag); "
                          f"a {floor.unit}-unit floor is not comparable (FLOOR-UNIT-1: refuse, do not warn)")


def analyze(runs_dir: str, control: str = "t48", floor: Optional[Floor] = None, expected_rows: int = EXPECTED_ROWS,
            allow_unscreened: bool = False, reps: int = 10000, seed: int = SEED) -> dict:
    check_floor(floor)
    recs = load_launches(runs_dir, expected_rows, allow_unscreened)
    valid = [r for r in recs if r.status == "valid"]
    by_arm: dict[str, list[LaunchRec]] = {}
    for r in valid:
        by_arm.setdefault(r.arm, []).append(r)
    if control not in by_arm or len(by_arm[control]) < 2:
        raise UnitRefusal(f"control arm {control!r} has {len(by_arm.get(control, []))} valid launch(es); "
                          f"need >= 2 to estimate a between-launch sd -- refusing to compare")
    arms = {a: summarize([r.tps for r in sorted(rs, key=lambda x: x.round_no)]) for a, rs in sorted(by_arm.items())}
    ctl = arms[control]
    pairs = {}
    for a, s in arms.items():
        if a == control:
            continue
        lo, hi = bootstrap_effect_pct(s["values"], ctl["values"], reps, seed)
        eff = (s["mean"] / ctl["mean"] - 1.0) * 100.0
        sd_p = pooled_sd(s, ctl)
        m_obs = mde_pct(sd_p, s["n"], ctl["n"], ctl["mean"]) if sd_p is not None else None
        m_floor = None
        if floor is not None:
            m_floor = mde_pct(floor.sd_pct / 100.0 * ctl["mean"], s["n"], ctl["n"], ctl["mean"])
        pairs[a] = dict(effect_pct=eff, boot_ci95_pct=(lo, hi), ci_excludes_zero=(lo > 0 or hi < 0),
                        pooled_sd=sd_p, mde_observed_pct=m_obs, mde_floor_pct=m_floor,
                        exceeds_mde_observed=(m_obs is not None and abs(eff) >= m_obs),
                        exceeds_mde_floor=(m_floor is not None and abs(eff) >= m_floor), n=s["n"], n_control=ctl["n"])
    return dict(unit="launch", control=control, floor=(None if floor is None else dict(sd_pct=floor.sd_pct, unit=floor.unit, n=floor.n)),
                alpha=ALPHA, power=POWER, bootstrap_reps=reps, seed=seed, arms=arms, vs_control=pairs,
                launches=[dict(label=r.label, arm=r.arm, round=r.round_no, tps=r.tps, n_dec=r.n_dec, status=r.status,
                               detail=r.detail) for r in recs],
                excluded=[dict(label=r.label, status=r.status, detail=r.detail) for r in recs if r.status != "valid"])


def render(res: dict) -> str:
    L = [f"R23-64 CPU decode thread sweep  unit=LAUNCH  control={res['control']}  "
         f"alpha={res['alpha']} power={res['power']}  bootstrap={res['bootstrap_reps']} seed={res['seed']}"]
    if res["floor"]:
        L.append(f"floor: sd {res['floor']['sd_pct']}%  unit={res['floor']['unit']}  n={res['floor']['n']}  (admissible: launch unit)")
    else:
        L.append("floor: none supplied (MDE is from the observed between-launch sd only)")
    L.append("")
    L.append(f"{'arm':<6}{'n':>3}{'mean t/s':>10}{'sd':>8}{'CV%':>7}{'95% CI (t)':>20}{'min..max':>16}")
    for a, s in res["arms"].items():
        ci = f"{s['ci95'][0]:.3f}..{s['ci95'][1]:.3f}" if s["ci95"] else "n/a"
        sd = f"{s['sd']:.3f}" if s["sd"] is not None else "n/a"
        cv = f"{s['cv_pct']:.2f}" if s["cv_pct"] is not None else "n/a"
        L.append(f"{a:<6}{s['n']:>3}{s['mean']:>10.3f}{sd:>8}{cv:>7}{ci:>20}{s['min']:>8.2f}..{s['max']:.2f}")
    L.append("")
    L.append(f"vs {res['control']} (effect % = arm/control - 1; CI = bootstrap over launches)")
    for a, p in res["vs_control"].items():
        mo = f"{p['mde_observed_pct']:.2f}%" if p["mde_observed_pct"] is not None else "n/a"
        mf = f"{p['mde_floor_pct']:.2f}%" if p["mde_floor_pct"] is not None else "n/a"
        v = ("RESOLVED" if p["ci_excludes_zero"] and p["exceeds_mde_observed"] else "UNRESOLVED")
        L.append(f"  {a:<5} {p['effect_pct']:+7.2f}%  CI [{p['boot_ci95_pct'][0]:+.2f}, {p['boot_ci95_pct'][1]:+.2f}]%  "
                 f"MDE(observed) {mo}  MDE(floor) {mf}  n={p['n']}v{p['n_control']}  {v}")
    if res["excluded"]:
        L.append("")
        L.append("EXCLUDED launches (never entered a statistic):")
        for e in res["excluded"]:
            L.append(f"  {e['label']:<10} {e['status']}")
    return "\n".join(L)


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--runs", required=True)
    ap.add_argument("--control", default="t48")
    ap.add_argument("--floor-sd-pct", type=float, default=None)
    ap.add_argument("--floor-unit", default=None, help="launch|process|session are admissible; 'arm' is REFUSED")
    ap.add_argument("--floor-n", type=int, default=None)
    ap.add_argument("--expected-rows", type=int, default=EXPECTED_ROWS)
    ap.add_argument("--allow-unscreened", action="store_true", help="keep launches with no sampler records (default: exclude)")
    ap.add_argument("--bootstrap", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)
    floor = None
    if a.floor_sd_pct is not None or a.floor_unit is not None:
        if a.floor_sd_pct is None:
            print("REFUSED: --floor-unit without --floor-sd-pct", file=sys.stderr)
            return 2
        floor = Floor(a.floor_sd_pct, a.floor_unit, a.floor_n)
    try:
        res = analyze(a.runs, a.control, floor, a.expected_rows, a.allow_unscreened, a.bootstrap, a.seed)
    except UnitRefusal as e:
        print(f"REFUSED (FLOOR-UNIT-1): {e}", file=sys.stderr)
        return 2
    print(render(res))
    if a.json:
        with open(a.json, "w") as fh:
            json.dump(res, fh, indent=1, sort_keys=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
