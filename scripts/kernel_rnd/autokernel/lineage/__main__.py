"""CLI: python3 -m scripts.kernel_rnd.autokernel.lineage <command>

  generate  --repo LLAMA --root EPYC_ROOT --out DIR [--staging DIR] [--date YYYYMMDD]
            writes ledger.json, fold-list.json, fold-list.md, presence-gate.json and
            (with --staging) unpublished Spec-D row specs. Exit 3 if any feature was
            refused (the list is still written; refusals are listed, never emitted).
  check-fold --ledger ledger.json TAG...   exit 0 if every TAG may fold, 3 if refused.
  check-presence --repo LLAMA --manifest presence-gate.json --candidate REF
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
from pathlib import Path
import sys

from . import ledger as L

HANDOFFS = ("handoffs/active/autokernel-akx-program.md",
            "handoffs/active/kernel-champion-and-promotion.md")
MANIFEST_DIR = "docs/reference/kernel-manifests"


def _read_sources(root: Path) -> tuple[dict[str, str], dict[str, str]]:
    dnf = {h: (root / h).read_text(encoding="utf-8") for h in HANDOFFS if (root / h).exists()}
    manifests = {f"{MANIFEST_DIR}/{p.name}": p.read_text(encoding="utf-8")
                 for p in sorted((root / MANIFEST_DIR).glob("llama-*-manifest.md"))}
    return dnf, manifests


def _dump(path: Path, body) -> None:
    path.write_text(json.dumps(body, indent=1, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="lineage")
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--repo", type=Path, default=Path("/mnt/raid0/llm/llama.cpp"))
    g.add_argument("--root", type=Path, default=Path("/workspace"))
    g.add_argument("--out", type=Path, required=True)
    g.add_argument("--staging", type=Path)
    g.add_argument("--champion", default=L.CHAMPION_BRANCH)
    g.add_argument("--current-fold")
    g.add_argument("--production", default=L.PRODUCTION_ANCHOR)
    g.add_argument("--window", type=int, default=3,
                   help="read side branches c(N-window)..cN (default 3: C14-C17 for C18)")
    g.add_argument("--verdicts", type=Path, help="JSON {row_id: verdict} (P1 output)")
    g.add_argument("--date", default=_dt.date.today().strftime("%Y%m%d"))
    c = sub.add_parser("check-fold")
    c.add_argument("--ledger", type=Path, required=True)
    c.add_argument("tags", nargs="+")
    p = sub.add_parser("check-presence")
    p.add_argument("--repo", type=Path, default=Path("/mnt/raid0/llm/llama.cpp"))
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--candidate", required=True)
    a = ap.parse_args(argv)

    if a.cmd == "check-fold":
        feats = {f["tag"]: f for f in json.loads(a.ledger.read_text())["features"]}
        rc = 0
        for t in a.tags:
            try:
                L.fold_entry(feats[t]) if t in feats else (_ for _ in ()).throw(
                    L.FoldRefused(f"{t}: not in ledger"))
                print(f"OK {t}")
            except L.FoldRefused as exc:
                print(f"REFUSED {exc}")
                rc = 3
        return rc
    if a.cmd == "check-presence":
        res = L.check_presence(L.git_runner(a.repo), json.loads(a.manifest.read_text()),
                               a.candidate)
        print(json.dumps(res, indent=1))
        return 0 if res["passed"] else 1

    dnf, manifests = _read_sources(a.root)
    verdicts = json.loads(a.verdicts.read_text()) if a.verdicts else None
    led = L.build_ledger(L.git_runner(a.repo), champion=a.champion, production=a.production,
                         current_fold=a.current_fold, dnf_texts=dnf, manifests=manifests,
                         verdicts=verdicts, window=a.window)
    folds = L.fold_list(led)
    pres = L.presence_manifest(led, folds)
    a.out.mkdir(parents=True, exist_ok=True)
    _dump(a.out / "ledger.json", led)
    _dump(a.out / "fold-list.json", folds)
    (a.out / "fold-list.md").write_text(L.fold_markdown(folds), encoding="utf-8")
    _dump(a.out / "presence-gate.json", pres)
    staged = []
    if a.staging:
        staged = L.write_staged(L.specd_rows(folds, date=a.date,
                                             dnf_shas=[d["sha"] for d in led["do_not_fold"]]),
                                a.staging)
    print(f"{folds['candidate']}: {len(folds['entries'])} fold entries, "
          f"{len(folds['refused'])} refused; presence entries {len(pres['entries'])}, "
          f"defects {len(pres['defects'])}; staged Spec-D rows {len(staged)} (NOT published)")
    return 3 if folds["refused"] else 0


if __name__ == "__main__":
    sys.exit(main())
