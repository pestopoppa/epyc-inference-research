"""AK-SURFACE-SERVING-1 (follow-up): two-case measurements as the evaluator for serving-knob candidates.

A serving-knob arm is scored on BOTH served measurements, parsed from the rows' own output shapes:

  * BEST case, steady state: row `ak-27b-served-steady-state-gpu-20261124` (job.py there): one long-lived
    np4 server, closed-loop K = 1/2/4 in flight, `results.jsonl` units `{kind: arm_run|dropped, label, valid,
    cells: {"1"|"2"|"4": cell}, probe, info}`. NOT FILED YET: this parser is written from job.py's shape.
  * WORST case, burst: row `ak-27b-served-np4-floor-gpu-20261124` (4 simultaneous cold 3.8k prompts, 384
    tokens, a fresh server per launch), `out/results.jsonl` launch units. Also the served noise floor.

`screen_cases` ranks by best-case aggregate tok/s and rejects an arm on a worst-case guard rail (fits memory,
ITL p95 <= +20% over control, correct output). Both cases carry tokens_per_verify_step =
predicted_n / (predicted_n - draft_n_accepted), and each arm is decomposed against control into a drafter
effect (tokens per step) and a step-cost effect (verify steps per second) so the two stay separate.
SCREENING ONLY, never a gate. Pure: no launches, no I/O beyond the parsers' file reads.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
import re
from pathlib import Path
import statistics
from typing import Any, Iterable, Mapping

from .serving_knobs import CONTROL, ITL_REGRESSION_MAX, SCREEN_SCHEMA, ScreeningInvalid, vram_capacity_gib

SCHEMA = "epyc.autokernel.serving_knob_measure.v1"
KS = (1, 2, 4)
#: steady-state units per label; a label's cells are the MEDIAN over its valid units (one K=4 unit is one draw).
_GIB = float(1 << 30)
_MEMORY_FAILURE = re.compile(r"out ?of ?memory|\boom\b|not-resident|failed to (allocate|create context|load model)", re.I)
_CELL_KEYS = ("aggregate_tok_s", "per_request_decode_tok_s_p50", "tokens_per_verify_step", "ttft_p95_s", "itl_p95_s")


def tokens_per_verify_step(predicted_n: float, draft_n_accepted: float) -> float | None:
    """predicted_n / (predicted_n - draft_n_accepted): tokens emitted per target verify step. None when undefined."""
    return predicted_n / (predicted_n - draft_n_accepted) if predicted_n - draft_n_accepted > 0 else None


def _pct(values: Iterable[float], q: float) -> float | None:
    """Linear-interpolated percentile, the same definition the steady-state job uses."""
    v = sorted(values)
    if not v:
        return None
    k = (len(v) - 1) * q
    lo, hi = int(k), min(len(v) - 1, int(k) + 1)
    return v[lo] + (v[hi] - v[lo]) * (k - lo)


def _gib(res: Any) -> float | None:
    """`own_vram_b` of a residency record in GiB (the one unit this module screens in); None = no info."""
    b = res.get("own_vram_b") if isinstance(res, Mapping) else None
    return b / _GIB if isinstance(b, int) and not isinstance(b, bool) else None


def _num(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) else None


@dataclass(frozen=True)
class BestCase:
    """One arm's steady-state measurement. `cells` is K -> {aggregate_tok_s, per_request_decode_tok_s_p50,
    tokens_per_verify_step, ttft_p95_s, itl_p95_s}; `fits` is the job's startup probe (ready, alive, no OOM line)."""
    cells: Mapping[int, Mapping[str, float | None]]
    valid: bool
    invalid_why: tuple[str, ...] = ()
    fits: bool = True
    fit_why: str | None = None
    probe_garbage: str | None = None
    own_vram_gib: float | None = None
    n_units: int = 0                          # valid units the cells' medians are taken over

    @property
    def k4_over_k1(self) -> float | None:
        a, b = (self.cells.get(k, {}).get("aggregate_tok_s") for k in (1, 4))
        return b / a if a and b else None


@dataclass(frozen=True)
class BurstLaunch:
    """One fresh-server launch of the burst row: 4 concurrent cold prompts."""
    valid: bool
    why: str                                  # "ok" or the failure the job recorded
    aggregate_tok_s: float | None = None
    predicted_n: int = 0
    draft_n_accepted: int = 0
    decode_ms_per_token: tuple[float, ...] = ()   # per request: predicted_ms / predicted_n
    own_vram_gib: float | None = None
    invalid_why: tuple[str, ...] = ()


@dataclass(frozen=True)
class WorstCase:
    launches: tuple[BurstLaunch, ...]

    @property
    def good(self) -> tuple[BurstLaunch, ...]:
        return tuple(x for x in self.launches if x.valid and x.aggregate_tok_s)

    @property
    def aggregate_tok_s(self) -> float | None:
        return statistics.median(x.aggregate_tok_s for x in self.good) if self.good else None

    @property
    def tokens_per_verify_step(self) -> float | None:
        return tokens_per_verify_step(sum(x.predicted_n for x in self.good), sum(x.draft_n_accepted for x in self.good))

    @property
    def itl_p95_ms(self) -> float | None:
        """p95 over requests of the per-request mean decode ms/token (burst records carry no per-event gaps, so this
        is the closest available ITL statistic; it spans decode under prefill interference)."""
        return _pct([m for x in self.good for m in x.decode_ms_per_token], 0.95)


@dataclass(frozen=True)
class ArmMeasurement:
    arm_id: str
    best: BestCase | None = None
    worst: WorstCase | None = None


# --- parsers -----------------------------------------------------------------------------
def _jsonl(source: Path | str | Iterable[Mapping]) -> list[dict]:
    if isinstance(source, (str, Path)):
        return [json.loads(line) for line in Path(source).read_text(encoding="utf-8").splitlines() if line.strip()]
    return [dict(x) for x in source]


def parse_burst_results(source: Path | str | Iterable[Mapping]) -> WorstCase:
    """`out/results.jsonl` of the burst row -> WorstCase (all launches kept, valid or not). The file's `arm` field is
    the A/A label of the floor design, not a knob arm; the caller attaches the knob arm id."""
    launches = []
    for u in _jsonl(source):
        rec = u.get("rec") or {}
        reqs = rec.get("requests") or []
        res = u.get("res_post") or u.get("res") or {}
        decode = tuple(r["predicted_ms"] / r["predicted_n"] for r in reqs
                       if _num(r.get("predicted_ms")) and _num(r.get("predicted_n")))
        launches.append(BurstLaunch(
            valid=bool(u.get("valid")), why=str(u.get("why", "ok")), aggregate_tok_s=_num(rec.get("aggregate_tok_s")),
            predicted_n=sum(int(r.get("predicted_n") or 0) for r in reqs),
            draft_n_accepted=sum(int(r.get("draft_n_accepted") or 0) for r in reqs), decode_ms_per_token=decode,
            own_vram_gib=_gib(res), invalid_why=tuple(str(x) for x in (u.get("invalid_why") or ()))))
    if not launches:
        raise ScreeningInvalid("burst results are empty")
    return WorstCase(tuple(launches))


def parse_burst_floor(result: Mapping[str, Any]) -> dict:
    """The served noise floor from the burst row's RESULT.json (`floor`): unit=process A/A pair scatter."""
    floor = result.get("floor") if isinstance(result, Mapping) else None
    keys = ("sd_ln_ratio", "screen_floor_pct", "serving_floor_pct", "n_pairs")
    if not isinstance(floor, Mapping) or any(_num(floor.get(k)) is None for k in keys) or floor.get("unit") != "process":
        raise ScreeningInvalid("RESULT.json carries no unit=process floor with sd/screen/serving/n")
    return {k: floor[k] for k in keys} | {"unit": "process", "verdict": result.get("verdict")}


def _median_cells(units: list[Mapping]) -> dict[int, dict]:
    ks = {int(k) for u in units for k in (u.get("cells") or {})}
    out = {}
    for k in sorted(ks):
        cs = [u["cells"][str(k)] for u in units if str(k) in (u.get("cells") or {})]
        out[k] = {key: statistics.median(v) if (v := [x for c in cs if (x := _num(c.get(key))) is not None]) else None for key in _CELL_KEYS}
    return out


def _best_case(units: list[Mapping]) -> BestCase:
    drop = next((u for u in units if u.get("kind") == "dropped"), None)
    if drop is not None:                       # an arm that did not fit
        return BestCase({}, False, (str(drop.get("reason")),), fits=False, fit_why=str(drop.get("reason")))
    infos = [u.get("info") or {} for u in units]
    info_ok = all(infos)                       # a unit with no info proves nothing about fit: never fits=True
    fits = info_ok and all(i.get("ready") and i.get("alive") and not i.get("oom_line") for i in infos)
    vram = [_gib(i.get("residency")) for i in infos]
    ok = [u for u in units if u.get("valid")]
    bad = [str(x) for u in units if not u.get("valid") for x in (u.get("invalid_why") or ())]
    return BestCase(
        _median_cells(ok), bool(ok), tuple(bad) if ok else tuple(bad) or ("no-valid-units",), fits=fits,
        fit_why=next((i["oom_line"] for i in infos if i.get("oom_line")), None)
        or (None if fits else "info-missing" if not info_ok else "server not ready/alive"),
        probe_garbage=next((g for u in units if (g := (u.get("probe") or {}).get("garbage"))), None),
        own_vram_gib=max(vram) if vram and None not in vram else None, n_units=len(ok))


def parse_steady_state(units: Path | str | Iterable[Mapping], arm_map: Mapping[str, str]) -> dict[str, BestCase]:
    """Steady-state `results.jsonl` units (or RESULT.json `units`) -> {knob arm id: BestCase}. `arm_map` maps the
    row's labels (A, B, B65K ...) to knob arm ids; an unmapped label raises, so a different -c (B65K) is never
    silently merged into B. A `dropped` unit is an arm that did not fit. Every valid unit of a label is kept and each
    cell is the MEDIAN over them (`n_units`), not the last unit."""
    by_label: dict[str, list[Mapping]] = {}
    for u in _jsonl(units):
        kind, label = u.get("kind"), u.get("label") or u.get("arm")
        if kind not in ("arm_run", "dropped") or u.get("smoke"):
            continue
        if label not in arm_map:
            raise ScreeningInvalid(f"steady-state label {label!r} has no arm mapping {sorted(arm_map)}")
        by_label.setdefault(label, []).append(u)
    return {arm_map[label]: _best_case(us) for label, us in by_label.items()}


def measurements(best: Mapping[str, BestCase], worst: Mapping[str, WorstCase]) -> dict[str, ArmMeasurement]:
    """Join the two cases by knob arm id (an arm in only one case stays, and is rejected by the screen)."""
    return {a: ArmMeasurement(a, best.get(a), worst.get(a)) for a in sorted({*best, *worst})}


# --- evaluator ---------------------------------------------------------------------------
def _reasons(m: ArmMeasurement, ctrl_itl: float | None, *, rank_k: int, min_launches: int, itl_max: float,
             vram_cap_gib: float, min_units: int) -> list[str]:
    r: list[str] = []
    b, w = m.best, m.worst
    if b is None:
        r.append("best-case-missing")
    else:
        if not b.fits:
            r.append("info-missing" if b.fit_why == "info-missing" else "does-not-fit-memory")
        elif b.own_vram_gib is None:
            r.append("info-missing")             # no memory info is never a fit
        if b.own_vram_gib is not None and b.own_vram_gib > vram_cap_gib:
            r.append("does-not-fit-memory")
        if b.valid and b.n_units < min_units:
            r.append(f"insufficient-steady-units:{b.n_units}<{min_units}")
        if b.probe_garbage:
            r.append("incorrect-output")
        if not b.valid and b.fits:
            r.append("best-case-invalid")
        if b.fits and not (b.cells.get(rank_k) or {}).get("aggregate_tok_s"):
            r.append(f"best-case-k{rank_k}-unmeasured")
    if w is None:
        r.append("worst-case-missing")
    else:
        for x in w.launches:
            if x.valid:
                continue
            oom = bool(_MEMORY_FAILURE.search(x.why + " " + " ".join(x.invalid_why)))
            r.append("does-not-fit-memory" if oom else "incorrect-output" if x.why == "ok" else f"launch-failed:{x.why[:40]}")
        if any(x.own_vram_gib is not None and x.own_vram_gib > vram_cap_gib for x in w.launches):
            r.append("does-not-fit-memory")
        if any(x.valid and x.own_vram_gib is None for x in w.launches):
            r.append("info-missing")
        if len(w.good) < min_launches:
            r.append(f"insufficient-burst-launches:{len(w.good)}<{min_launches}")
        itl = w.itl_p95_ms
        if ctrl_itl is not None:
            if itl is None:
                r.append("itl-unmeasured")
            elif itl > ctrl_itl * (1 + itl_max):
                r.append(f"itl-regression>{itl_max * 100:g}%")
    return sorted(set(r))


def _ratio(a: float | None, b: float | None) -> float | None:
    return a / b if a is not None and b else None


def _decompose(agg: float | None, tps: float | None, c_agg: float | None, c_tps: float | None) -> dict:
    """aggregate ratio = tokens-per-step ratio (drafter effect) x verify-steps-per-second ratio (step-cost effect)."""
    steps = agg / tps if agg is not None and tps else None
    c_steps = c_agg / c_tps if c_agg is not None and c_tps else None
    return {"aggregate_ratio": _ratio(agg, c_agg), "tokens_per_step_ratio": _ratio(tps, c_tps),
            "steps_per_s": steps, "steps_per_s_ratio": _ratio(steps, c_steps)}


def _entry(m: ArmMeasurement, ctrl: ArmMeasurement, rank_k: int, ctrl_itl: float | None) -> dict:
    b, w = m.best, m.worst
    cb, cw = ctrl.best, ctrl.worst
    cell = (b.cells.get(rank_k) or {}) if b else {}
    ccell = (cb.cells.get(rank_k) or {}) if cb else {}
    out: dict[str, Any] = {"arm_id": m.arm_id, "best": None, "worst": None}
    if b:
        out["best"] = {"rank_k": rank_k, "aggregate_tok_s": cell.get("aggregate_tok_s"), "k4_over_k1": b.k4_over_k1,
                       "by_k": {str(k): dict(v) for k, v in sorted(b.cells.items())},
                       "tokens_per_verify_step": cell.get("tokens_per_verify_step"), "itl_p95_s": cell.get("itl_p95_s"),
                       "vs_control_pct": None if not ccell.get("aggregate_tok_s") or not cell.get("aggregate_tok_s")
                       else round(100 * (cell["aggregate_tok_s"] / ccell["aggregate_tok_s"] - 1), 3),
                       "decomposition": _decompose(cell.get("aggregate_tok_s"), cell.get("tokens_per_verify_step"),
                                                   ccell.get("aggregate_tok_s"), ccell.get("tokens_per_verify_step"))}
    if w:
        itl = w.itl_p95_ms
        out["worst"] = {"aggregate_tok_s": w.aggregate_tok_s, "n_launches": len(w.good),
                        "tokens_per_verify_step": w.tokens_per_verify_step, "itl_p95_ms": itl,
                        "itl_regression_pct": None if itl is None or not ctrl_itl else round(100 * (itl / ctrl_itl - 1), 3),
                        "decomposition": _decompose(w.aggregate_tok_s, w.tokens_per_verify_step,
                                                    cw.aggregate_tok_s if cw else None,
                                                    cw.tokens_per_verify_step if cw else None)}
    return out


def screen_cases(arms: Mapping[str, ArmMeasurement], *, control: str = CONTROL, rank_k: int = 4,
                 itl_regression_max: float = ITL_REGRESSION_MAX, min_burst_launches: int = 3,
                 min_steady_units: int = 3, vram_capacity_gib: float,
                 noise_floor: Mapping[str, Any] | None = None) -> dict:
    """Rank by best-case aggregate tok/s at K=`rank_k`; reject an arm failing a worst-case guard rail.

    The `control` arm (the served shape) needs BOTH cases, sound, because ITL is relative to it. `noise_floor`
    (from `parse_burst_floor`) is attached, and `best_arm` is named only when the top arm clears its
    `screen_floor_pct` (|vs_control_pct| >= floor; no floor means no best arm). `vram_capacity_gib` is required:
    take it from the registry (`serving_knobs.vram_capacity_gib`), there is no default. Screening only."""
    if control not in arms:
        raise ScreeningInvalid(f"no {control!r} arm: the ITL guard has nothing to be relative to")
    if rank_k not in KS or not 0 <= itl_regression_max < 1 or min_burst_launches < 1 or min_steady_units < 1 \
            or not (isinstance(vram_capacity_gib, (int, float)) and vram_capacity_gib > 0):
        raise ScreeningInvalid("guard parameters out of range")
    ctrl = arms[control]
    if ctrl.best is None or ctrl.worst is None:
        raise ScreeningInvalid("control arm needs both the best case and the worst case")
    ctrl_itl = ctrl.worst.itl_p95_ms
    bad = _reasons(ctrl, None, rank_k=rank_k, min_launches=min_burst_launches, itl_max=itl_regression_max,
                   vram_cap_gib=vram_capacity_gib, min_units=min_steady_units)
    if bad or ctrl_itl is None or not (ctrl.best.cells.get(rank_k) or {}).get("aggregate_tok_s"):
        raise ScreeningInvalid(f"control arm is unsound: {bad or ['itl/throughput-unmeasured']}")
    kept, rejected = [], []
    for arm_id in sorted(a for a in arms if a != control):
        m = arms[arm_id]
        reasons = _reasons(m, ctrl_itl, rank_k=rank_k, min_launches=min_burst_launches, itl_max=itl_regression_max,
                           vram_cap_gib=vram_capacity_gib, min_units=min_steady_units)
        e = _entry(m, ctrl, rank_k, ctrl_itl) if m.best and m.worst and not reasons else {"arm_id": arm_id}
        (rejected if reasons else kept).append(e | {"reasons": reasons} if reasons else e)
    ranking = sorted([_entry(ctrl, ctrl, rank_k, ctrl_itl) | {"is_control": True}, *kept],
                     key=lambda e: (-e["best"]["aggregate_tok_s"], not e.get("is_control"), e["arm_id"]))
    for i, e in enumerate(ranking, 1):
        e["rank"] = i
    top = ranking[0]
    floor = _num((noise_floor or {}).get("screen_floor_pct"))
    vs = (top.get("best") or {}).get("vs_control_pct")
    why = ("control-leads" if top.get("is_control") else "noise-floor-missing" if floor is None
           else "below-screen-floor" if vs is None or abs(vs) < floor else "clears-screen-floor")
    best_arm = top["arm_id"] if why == "clears-screen-floor" else None
    return {"schema": SCREEN_SCHEMA, "screening_only": True, "promotable": False, "control": control,
            "rank_metric": f"best-case aggregate_tok_s at K={rank_k}",
            "guards": {"scope": "worst-case burst", "itl_p95_regression_max": itl_regression_max, "fit_memory": True, "correct_output": True,
                       "min_burst_launches": min_burst_launches, "min_steady_units": min_steady_units, "vram_capacity_gib": vram_capacity_gib},
            "burst_noise_floor": dict(noise_floor) if noise_floor else None, "ranking": ranking, "rejected": rejected,
            "best_arm": best_arm, "best_arm_why": why,
            "note": "screen, not a gate: arms differ in np/batch/drafter; confirm a winner with a paired A/B"}


def screen_stage(*, server: str, steady_results: Path | str | Iterable[Mapping], arm_map: Mapping[str, str],
                 burst_results: Mapping[str, Path | str | Iterable[Mapping]], floor_result: Path | str | Mapping,
                 registry: Any = None, **guards: Any) -> dict:
    """The chain-stage entry point (pure; file reads only): parse the two rows' outputs, take the fit bound from the
    registry (`serving_knobs.vram_capacity_gib`), screen once. `burst_results` maps knob arm id -> that arm's burst
    `results.jsonl`; `floor_result` is the burst row's RESULT.json (path or parsed). Raises ScreeningInvalid /
    KnobError rather than returning a partial screen."""
    floor_body = json.loads(Path(floor_result).read_text(encoding="utf-8")) if isinstance(floor_result, (str, Path)) else floor_result
    return screen_cases(
        measurements(parse_steady_state(steady_results, arm_map),
                     {arm: parse_burst_results(src) for arm, src in burst_results.items()}),
        vram_capacity_gib=vram_capacity_gib(server, registry), noise_floor=parse_burst_floor(floor_body), **guards)


__all__ = ["ArmMeasurement", "BestCase", "BurstLaunch", "WorstCase", "measurements", "parse_burst_floor",
           "parse_burst_results", "parse_steady_state", "screen_cases", "screen_stage", "tokens_per_verify_step"]
