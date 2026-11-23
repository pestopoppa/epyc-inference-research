"""Derive per-recipe CPU accumulator bundles from kept experiment rows (AKU-17).

Reads ``experiments.db`` read-only and projects each selected ``kept`` row's serving A/B
(`comparison`) into a ``cpu_keep_bundle`` keep. Nothing is invented: effect, floor, floor
unit, contention and statistic come from the row, a row missing any of them refuses, and the
magnitude is ``provisional`` unless ``cpu_keep_bundle`` itself allows ``settled``.
"""
from __future__ import annotations

import json
import re
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from . import cpu_keep_bundle as ckb

_ATTEMPT = re.compile(r"attempt_id`? `([0-9a-f]{12})\.\.\.`")


def attempt_prefixes_in_manifest(text: str) -> list[str]:
    """Attempt-id prefixes a kernel manifest cites as campaign keep records (its fold set)."""
    return sorted(set(_ATTEMPT.findall(text)))


def _keep_from_row(attempt_id: str, mechanism: str, payload: Mapping) -> tuple[str, str, dict]:
    cmp = payload.get("comparison") or {}
    need = ("effect_pct", "noise_floor_pct", "floor_unit", "recipe", "recipe_hash")
    missing = [k for k in need if cmp.get(k) is None]
    estimator = (cmp.get("measurement_plan") or {}).get("estimator")
    contention = (cmp.get("residency") or {}).get("contention")
    if not estimator:
        missing.append("measurement_plan.estimator")
    if not contention:
        missing.append("residency.contention")
    if missing:
        raise ckb.CpuBundleError(f"{attempt_id[:12]}: row lacks {missing}")
    keep = {"keep_id": f"{mechanism}@{attempt_id[:12]}", "effect_pct": cmp["effect_pct"],
            "magnitude": "provisional", "contention_label": contention, "statistic": estimator,
            "floor_pct": cmp["noise_floor_pct"], "floor_unit": cmp["floor_unit"]}
    return cmp["recipe"], cmp["recipe_hash"], keep


def derive(db: Path | str | Sequence[Path | str], prefixes: Iterable[str], *, champion_of_record: str,
           tip: str, skipped: list | None = None) -> list[dict]:
    """One validated bundle per recipe for the kept rows whose attempt id starts with a prefix.

    ``db`` may be several stores (one fold cites keeps from several campaigns); a prefix matching
    kept rows in two stores refuses as ambiguous. A prefix that matches no kept row refuses, unless the caller passes ``skipped`` (a list that
    then collects them): a fold manifest cites keeps from several campaign stores, and the ones
    this store lacks are reported, never silently dropped."""
    rows = []
    for one in ([db] if isinstance(db, (str, Path)) else db):
        with closing(sqlite3.connect(f"file:{one}?mode=ro", uri=True)) as con:
            rows += con.execute("select attempt_id, mechanism_id, payload from experiments "
                                "where status='kept' order by recorded_at").fetchall()
    groups: dict[tuple[str, str], list[dict]] = {}
    for prefix in sorted(set(prefixes)):
        hit = [r for r in rows if r[0].startswith(prefix)]
        if not hit and skipped is not None:
            skipped.append(prefix)
            continue
        if len(hit) != 1:
            raise ckb.CpuBundleError(f"prefix {prefix!r} matches {len(hit)} kept rows")
        aid, mech, payload = hit[0]
        recipe, rhash, keep = _keep_from_row(aid, mech, json.loads(payload))
        groups.setdefault((recipe, rhash), []).append(keep)
    # The recipe NAME is reused across recipe revisions (DS41: two hashes under one name), so the
    # bundle is keyed name.hash8; keying by name alone made the second bundle overwrite the first.
    return [ckb.build(recipe=f"{r}.{h[:8]}", recipe_hash=h, champion_of_record=champion_of_record, tip=tip,
                      keeps=keeps) for (r, h), keeps in sorted(groups.items())]


def record(db: Path | str | Sequence[Path | str], prefixes: Iterable[str], out_dir: Path | str, *,
           champion_of_record: str, tip: str, skipped: list | None = None) -> list[Path]:
    return [ckb.write(Path(out_dir), b) for b in
            derive(db, prefixes, champion_of_record=champion_of_record, tip=tip, skipped=skipped)]


def main(argv: list[str] | None = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--db", required=True, action="append", help="experiments.db; repeat for several stores")
    ap.add_argument("--manifest", required=True, help="kernel manifest whose cited keeps are the fold")
    ap.add_argument("--out", required=True)
    ap.add_argument("--champion-of-record", required=True)
    ap.add_argument("--tip", required=True)
    a = ap.parse_args(argv)
    prefixes = attempt_prefixes_in_manifest(Path(a.manifest).read_text(encoding="utf-8"))
    skipped: list = []
    for path in record(a.db, prefixes, a.out, champion_of_record=a.champion_of_record, tip=a.tip,
                       skipped=skipped):
        print(path)
    print(f"not in this store (skipped): {skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
