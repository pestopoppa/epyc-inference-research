"""Per-surface accumulator fire decision (AKU-18) -- the data contract the /loop card reads.

One record, ``epyc.autokernel.accumulator_surfaces.v1``, holds a section per surface (cpu, gpu).
Each section carries the surface's fire decision and its compounded figure, and the figure is
labelled by what it is:

* ``measured``  -- the bundle's own ``current_snapshot`` re-measurement of tip vs champion of record;
* ``estimate``  -- anything else, in particular the product of per-keep solo effects, which is never
  a measurement of the bundle. The label text is ``ESTIMATE`` and the reader refuses an
  ``estimate`` without it.

The decision mirrors ``accumulate.gate_trigger`` and, like it, fails closed: no calibrated
serving floor, or a stale baseline, is ``cannot_decide`` rather than a guess. The threshold
trigger is honoured only for a ``measured`` figure; an estimate can fire only on the mandatory
cadence. The AKU-17 per-recipe CPU fold bundles ride along as ``fold_estimates`` (always
ESTIMATE, with their provisional count).
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Mapping, Sequence

from . import accumulate, cpu_keep_bundle, status

SCHEMA = "epyc.autokernel.accumulator_surfaces.v1"
FILENAME = "accumulator_surfaces.json"
ESTIMATE_LABEL = "ESTIMATE (product of solo keep effects, not a measurement of the bundle)"
MEASURED_LABEL = "MEASURED (current snapshot of tip vs champion of record)"


def fire_decision(*, n_keeps: int, validity: str, pct: float | None, since_gate: int,
                  floor_pct: float | None, policy: accumulate.AccumulatorPolicy) -> dict:
    if floor_pct is None:
        return {"decision": "cannot_decide", "trigger": None,
                "reason": "no calibrated serving floor: the gate could not judge a reading (fail-closed)"}
    if validity == accumulate.MEASUREMENT_STALE_BASELINE:
        return {"decision": "cannot_decide", "trigger": None, "reason": "stale baseline"}
    threshold = (validity == accumulate.MEASUREMENT_CURRENT and pct is not None
                 and pct >= policy.fire_threshold_pct(floor_pct))
    cadence = policy.every_keeps > 0 and since_gate >= policy.every_keeps
    trigger = "both" if threshold and cadence else "threshold" if threshold else \
        "cadence" if cadence else None
    if trigger:
        return {"decision": "fire", "trigger": trigger, "reason": f"{trigger} trigger holds"}
    why = "no keeps in the bundle" if n_keeps == 0 else (
        f"{since_gate}/{policy.every_keeps} keeps to the cadence; "
        + ("estimate cannot fire the threshold trigger" if validity != accumulate.MEASUREMENT_CURRENT
           else "measured figure below the threshold"))
    return {"decision": "accumulate", "trigger": None, "reason": why}


def _figure(bundle: Mapping) -> dict:
    n = len(bundle.get("keeps") or [])
    validity = bundle.get("measurement_validity") or "unknown_legacy"
    pct = bundle.get("compounded_bench_pct")
    if n == 0 or not isinstance(pct, (int, float)) or isinstance(pct, bool):
        return {"kind": "none", "pct": None, "label": "no keeps in the bundle"}
    if validity == accumulate.MEASUREMENT_CURRENT:
        return {"kind": "measured", "pct": float(pct), "label": MEASURED_LABEL}
    return {"kind": "estimate", "pct": float(pct), "label": ESTIMATE_LABEL}


def _fold_estimates(cpu_dir: Path | None) -> list[dict]:
    out = []
    for path in sorted(Path(cpu_dir).glob(cpu_keep_bundle.FILENAME_PREFIX + "*.json")) if cpu_dir else []:
        b = cpu_keep_bundle.read(path)
        out.append({"recipe": b["recipe"], "n_keeps": len(b["keeps"]),
                    "pct": b["compounded_bench_pct"], "label": ESTIMATE_LABEL,
                    "provisional_keeps": sum(k["magnitude"] == "provisional" for k in b["keeps"])})
    return out


def surface_section(surface: str, bundle_path: Path | str, *, floor_pct: float | None,
                    policy: accumulate.AccumulatorPolicy | None = None,
                    cpu_fold_dir: Path | str | None = None) -> dict:
    policy = policy or accumulate.AccumulatorPolicy()
    bundle = json.loads(Path(bundle_path).read_text(encoding="utf-8"))
    n = len(bundle.get("keeps") or [])
    fig = _figure(bundle)
    since = bundle.get("keeps_since_serving_gate")
    since = since if isinstance(since, int) and not isinstance(since, bool) else 0
    validity = bundle.get("measurement_validity") or "unknown_legacy"
    return {"surface": surface, "source": str(bundle_path), "n_keeps": n,
            "measurement_validity": validity, "figure": fig, "floor_pct": floor_pct,
            "keeps_since_serving_gate": since, "every_keeps": policy.every_keeps,
            "fire_multiple": policy.fire_multiple,
            "fire": fire_decision(n_keeps=n, validity=validity, pct=fig["pct"], since_gate=since,
                                  floor_pct=floor_pct, policy=policy),
            "fold_estimates": _fold_estimates(Path(cpu_fold_dir)) if cpu_fold_dir and surface == "cpu" else []}


def build(sections: Sequence[dict], *, now: float | None = None) -> dict:
    ts = time.time() if now is None else now
    return {"schema": SCHEMA, "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts)),
            "surfaces": list(sections)}


def publish(out_dir: Path | str, record: Mapping) -> Path:
    return status.write_json(Path(out_dir), FILENAME, dict(record), prefix=".accumulator-surfaces-")


def main(argv: list[str] | None = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--surface", action="append", required=True, metavar="NAME=BUNDLE.json[:FLOOR_PCT]")
    ap.add_argument("--cpu-fold-dir")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    sections = []
    for spec in a.surface:
        name, _, rest = spec.partition("=")
        path, _, floor = rest.partition(":")
        sections.append(surface_section(name, path, floor_pct=float(floor) if floor else None,
                                        cpu_fold_dir=a.cpu_fold_dir))
    print(publish(a.out, build(sections)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
