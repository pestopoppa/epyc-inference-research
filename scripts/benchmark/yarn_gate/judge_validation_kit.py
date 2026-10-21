#!/usr/bin/env python3
"""YARN-JUDGE-1: the ~50-item hand-validation kit for the local judge. Stdlib only, no model.

Subcommands (run in order)::

    select  --source rows.jsonl [--n 50] [--seed 20261021] --out items.jsonl
    sheet   --items items.jsonl --out labels_sheet.csv          # BLIND: no judge verdict column
    judge   --items items.jsonl --port P --judge-model M --out judged.jsonl   # needs a served judge
    agree   --sheet labels_sheet_filled.csv --judged judged.jsonl --out agreement.json
    pin     --agreement agreement.json --judge-model M --out judge_pin.json

``select`` takes rows from any already-logged judged-type data (JSONL; a JSON list also accepted) with
``question`` / ``answer`` (reference) / ``output`` (candidate) fields, remappable with ``--map``. It
samples stratified by ``--strata-key`` (default ``task``), deterministic from ``--seed``, and
de-duplicates on the (question, output) hash. The labeller fills ``human_label`` (1 correct / 0 incorrect;
``?`` skips) in the sheet WITHOUT seeing the judge verdict.

``agree`` reports accuracy, Cohen kappa, the confusion matrix and a Wilson 95% interval, over items
that are both labelled and judged (unparsed judge rows are counted separately, never as 0). Judging
in ``judge`` imports ``local_judge`` and refuses inside a timing window. Acceptance for YARN-JUDGE-1 is
"agreement REPORTED with the pin", not a threshold; ``--min-kappa`` only adds a flag to the JSON.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

SHEET_FIELDS = ["item_id", "task", "question", "reference", "candidate", "human_label", "notes"]


def _read_rows(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8")
    if text.lstrip().startswith("["):
        return list(json.loads(text))
    return [json.loads(l) for l in text.splitlines() if l.strip()]


def _write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows), encoding="utf-8")


def _ref_text(x) -> str:
    return " | ".join(map(str, x)) if isinstance(x, (list, tuple)) else str(x)


def select_items(rows: list[dict], n: int = 50, seed: int = 20261021, strata_key: str = "task",
                 mapping: dict | None = None) -> list[dict]:
    m = {"question": "question", "reference": "answer", "candidate": "output", **(mapping or {})}
    seen, pool = set(), defaultdict(list)
    for r in rows:
        try:
            q, ref, cand = r[m["question"]], r[m["reference"]], r[m["candidate"]]
        except KeyError:
            continue
        if not str(cand).strip():
            continue  # an empty output tells the labeller nothing
        h = hashlib.sha256(f"{q}\x00{cand}".encode()).hexdigest()
        if h in seen:
            continue
        seen.add(h)
        pool[str(r.get(strata_key, "all"))].append(
            {"task": str(r.get(strata_key, "all")), "question": str(q), "reference": _ref_text(ref),
             "candidate": str(cand), "src_hash": h})
    rng = random.Random(seed)
    strata = sorted(pool)
    for s in strata:
        rng.shuffle(pool[s])
    out, i = [], 0
    while len(out) < n and any(pool[s] for s in strata):  # round-robin across strata
        s = strata[i % len(strata)]
        i += 1
        if pool[s]:
            out.append(pool[s].pop())
    for k, it in enumerate(out):
        it["item_id"] = f"j{k:03d}"
    if len(out) < n:
        sys.stderr.write(f"WARNING: only {len(out)} of {n} items available\n")
    return out


def write_sheet(items: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=SHEET_FIELDS, extrasaction="ignore")
        w.writeheader()
        for it in items:
            w.writerow({**it, "human_label": "", "notes": ""})


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def cohen_kappa(pairs: list[tuple[int, int]]) -> float | None:
    n = len(pairs)
    if n == 0:
        return None
    po = sum(a == b for a, b in pairs) / n
    pa1 = sum(a for a, _ in pairs) / n
    pb1 = sum(b for _, b in pairs) / n
    pe = pa1 * pb1 + (1 - pa1) * (1 - pb1)
    if pe == 1.0:
        return None  # no variance in either rater: kappa undefined, report as null
    return (po - pe) / (1 - pe)


def agreement(sheet_rows: list[dict], judged_rows: list[dict], min_kappa: float | None = None) -> dict:
    judge = {r["item_id"]: r for r in judged_rows}
    pairs, skipped_human, unparsed, missing = [], 0, 0, 0
    for r in sheet_rows:
        lab = str(r.get("human_label", "")).strip()
        if lab not in ("0", "1"):
            skipped_human += 1
            continue
        j = judge.get(r["item_id"])
        if j is None:
            missing += 1
        elif j.get("verdict") not in (0, 1):
            unparsed += 1
        else:
            pairs.append((int(lab), int(j["verdict"])))
    n = len(pairs)
    agree_k = sum(a == b for a, b in pairs)
    conf = {f"human{a}_judge{b}": sum(1 for x in pairs if x == (a, b)) for a in (0, 1) for b in (0, 1)}
    lo, hi = wilson(agree_k, n)
    kappa = cohen_kappa(pairs)
    out = {"n_compared": n, "n_agree": agree_k, "accuracy": agree_k / n if n else None,
           "accuracy_wilson95": [lo, hi], "cohen_kappa": kappa, "confusion": conf,
           "n_human_skipped": skipped_human, "n_judge_unparsed": unparsed, "n_judge_missing": missing,
           "human_positive_rate": (sum(a for a, _ in pairs) / n) if n else None,
           "judge_positive_rate": (sum(b for _, b in pairs) / n) if n else None}
    if min_kappa is not None:
        out["min_kappa"] = min_kappa
        out["meets_min_kappa"] = (kappa is not None and kappa >= min_kappa)
    return out


def build_pin_record(agreement_obj: dict, pin_fields: dict, sheet_sha256: str) -> dict:
    return {"schema": "epyc.yarn_judge_pin.v1", "judge_pin": pin_fields, "validation": agreement_obj,
            "sheet_sha256": sheet_sha256, "comparable_to_published": False,
            "note": "relative to this pinned local judge only"}


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("select")
    s.add_argument("--source", required=True)
    s.add_argument("--n", type=int, default=50)
    s.add_argument("--seed", type=int, default=20261021)
    s.add_argument("--strata-key", default="task")
    s.add_argument("--map", nargs="*", default=[], help="question=KEY reference=KEY candidate=KEY")
    s.add_argument("--out", required=True)
    sh = sub.add_parser("sheet")
    sh.add_argument("--items", required=True)
    sh.add_argument("--out", required=True)
    j = sub.add_parser("judge")
    j.add_argument("--items", required=True)
    j.add_argument("--port", type=int, required=True)
    j.add_argument("--judge-model", required=True)
    j.add_argument("--out", required=True)
    a = sub.add_parser("agree")
    a.add_argument("--sheet", required=True)
    a.add_argument("--judged", required=True)
    a.add_argument("--min-kappa", type=float)
    a.add_argument("--out", required=True)
    p = sub.add_parser("pin")
    p.add_argument("--agreement", required=True)
    p.add_argument("--sheet", required=True)
    p.add_argument("--judge-model", required=True)
    p.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    if args.cmd == "select":
        mp = dict(kv.split("=", 1) for kv in args.map)
        items = select_items(_read_rows(Path(args.source)), args.n, args.seed, args.strata_key, mp)
        _write_jsonl(Path(args.out), items)
        print(f"selected {len(items)} items -> {args.out}")
    elif args.cmd == "sheet":
        write_sheet(_read_rows(Path(args.items)), Path(args.out))
        print(f"sheet -> {args.out} (fill human_label with 1/0; do NOT look at judge output first)")
    elif args.cmd == "judge":
        from scripts.benchmark.yarn_gate.local_judge import JudgePin, LocalJudge, JudgeError
        lj = LocalJudge(args.port, JudgePin(model=args.judge_model))
        rows = []
        for it in _read_rows(Path(args.items)):
            try:
                r = lj.judge(it["question"], it["reference"], it["candidate"])
                rows.append({"item_id": it["item_id"], "verdict": r["verdict"], "raw": r["raw"][:300]})
            except JudgeError as exc:
                rows.append({"item_id": it["item_id"], "verdict": None, "raw": f"unavailable: {exc}"})
        _write_jsonl(Path(args.out), rows)
    elif args.cmd == "agree":
        with Path(args.sheet).open(newline="", encoding="utf-8") as f:
            sheet = list(csv.DictReader(f))
        res = agreement(sheet, _read_rows(Path(args.judged)), args.min_kappa)
        Path(args.out).write_text(json.dumps(res, indent=2, sort_keys=True) + "\n")
        print(json.dumps(res, indent=2, sort_keys=True))
    elif args.cmd == "pin":
        from scripts.benchmark.yarn_gate.local_judge import JudgePin
        rec = build_pin_record(json.loads(Path(args.agreement).read_text()),
                               JudgePin(model=args.judge_model).claim_fields(), _sha(Path(args.sheet)))
        Path(args.out).write_text(json.dumps(rec, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
