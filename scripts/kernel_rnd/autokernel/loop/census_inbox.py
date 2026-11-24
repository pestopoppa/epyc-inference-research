#!/usr/bin/env python3
"""census_inbox.py -- keep BWS-*.md stubs in the loop's <store>/inbox in step with the bandwidth census (RSL-R2).

    python3 -m autokernel.loop.census_inbox --census <census.json> --store <store> [--spool DIR] [--dry-run]

Called by epyc-root ak_runend.py after the census_update step. `plan` is pure (census + inbox listing + spool listing in,
install/remove/skip out); `apply` writes it. One stub per SEEDABLE surface (open, measured, ownerless:
`census_update.seedable_surfaces`), named like `write_stubs` names it (`BWS-<path>-<op>.md`) and holding the same section,
so the planner reads it under "Operator suggestions" (controller/inbox.py) with no further wiring.

* Dedupe on target_surface: a surface is NOT installed when any other inbox file already carries `target_surface: <key>`
  (the committed `NN-bw-surfaces-*.md` seed, an operator note) or a candidate in the spool has a hypothesis_id naming it.
* Remove: a stub whose surface is now `covered` or `ok` (or that another inbox file now carries) is deleted.
  Only files named `BWS-*.md` are ever touched, and a stub whose surface is absent from the census is left alone.
* Idempotent: a stub is rewritten only when its body differs, so a rerun on the same census changes nothing.
Never writes the census, the spool, or any committed seed file.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

from .census_update import seedable_surfaces, skey, surface_seed_id, surface_seed_section

SPOOL = os.environ.get("AKCHAIN_SPOOL") or "/mnt/raid0/llm/tmp/akchain-spool"
PREFIX = "BWS-"
DONE = ("covered", "ok")


def _slug(x: object) -> str:
    return re.sub(r"[^a-z0-9]+", "-", str(x).lower()).strip("-")


def _read(p: Path) -> str:
    try:
        return p.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return ""


def spool_hypotheses(spool: str | Path | None) -> set[str]:
    """Slugged hypothesis_id of every candidate record in the spool (an absent spool is empty)."""
    out: set[str] = set()
    for p in sorted(Path(spool).glob("*.json")) if spool and Path(spool).is_dir() else []:
        try:
            rec = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(rec, dict) and rec.get("hypothesis_id"):
            out.add(_slug(rec["hypothesis_id"]))
    return out


def plan(census: dict, inbox: str | Path, spool: str | Path | None = None) -> dict:
    """{"install": {name: body}, "remove": [name], "skipped": {surface_seed_id: "inbox"|"spool"}}. Pure: writes nothing."""
    texts = {p.name: _read(p) for p in sorted(Path(inbox).glob("*.md"))}
    hyps = spool_hypotheses(spool)
    by_id = {surface_seed_id(s): s for s in census.get("surfaces") or []}
    install: dict[str, str] = {}
    skipped: dict[str, str] = {}
    dup: set[str] = set()
    for sid, s in by_id.items():
        rx = re.compile(rf"target_surface: {re.escape(skey(s))}[ \t]*$", re.M)
        if any(rx.search(t) for n, t in texts.items() if n != sid + ".md"):
            dup.add(sid)
    for s in seedable_surfaces(census):
        sid, name = surface_seed_id(s), surface_seed_id(s) + ".md"
        body = "\n".join(surface_seed_section(s, census)) + "\n"
        if sid in dup:
            skipped[sid] = "inbox"
        elif any(_slug(sid) in h or _slug(skey(s)) == h for h in hyps) and name not in texts:
            skipped[sid] = "spool"
        elif texts.get(name) != body:
            install[name] = body
    remove = sorted(n for n in texts if n.startswith(PREFIX) and n[:-3] in by_id
                    and (by_id[n[:-3]]["status"] in DONE or n[:-3] in dup))
    return {"install": {k: v for k, v in install.items() if k not in remove}, "remove": remove, "skipped": skipped}


def apply(p: dict, inbox: str | Path) -> dict:
    """Write a plan. A missing inbox is an error (a mistyped store must not be created), never a silent mkdir."""
    inbox = Path(inbox)
    if not inbox.is_dir():
        raise FileNotFoundError(f"inbox directory {inbox} does not exist")
    for name, body in p["install"].items():
        tmp = inbox / ("." + name + ".tmp")  # dot-prefixed: controller/inbox.py skips it silently if a reader races
        tmp.write_text(body, encoding="utf-8")
        os.replace(tmp, inbox / name)
    for name in p["remove"]:
        (inbox / name).unlink(missing_ok=True)
    return {"written": len(p["install"]), "removed": len(p["remove"]), "skipped": p["skipped"]}


def sync(census_path: str | Path, store: str | Path, spool: str | Path | None = SPOOL, dry_run: bool = False) -> dict:
    inbox = Path(store) / "inbox"
    with open(census_path, encoding="utf-8") as f:
        p = plan(json.load(f), inbox, spool)
    if dry_run:
        return {"written": 0, "removed": 0, "would_write": sorted(p["install"]), "would_remove": p["remove"],
                "skipped": p["skipped"]}
    return apply(p, inbox)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--census", required=True)
    ap.add_argument("--store", required=True)
    ap.add_argument("--spool", default=SPOOL)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    try:
        rep = sync(a.census, a.store, a.spool, a.dry_run)
    except (OSError, ValueError, KeyError) as e:
        print("BW-INBOX " + json.dumps({"error": f"{type(e).__name__}: {e}"}))
        return 1
    print("BW-INBOX " + json.dumps(rep, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
