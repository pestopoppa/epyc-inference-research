#!/usr/bin/env python3
"""Seed the loop's information base from what we already knew.

    python3 -m autokernel.loop.seed --store /mnt/raid0/llm/autokernel/loop-memory

WHY THIS EXISTS
---------------
The loop shipped with two inbound channels and nothing in either of them. Run 6 spent five
iterations proposing `dp4a` and `quantize_q8_1` work that the critic rejected as "already
present in production v9" -- while `handoffs/active/mi210-q8-dequant-gemv-roofline.md` had
already MEASURED `quantize_q8_1` at 5.68% of decode and written the decisive experiment for
it. The planner was re-deriving, badly, what the backlog already held.

Two channels, and the difference matters:

  * ``inbox/`` -- hypotheses to explore. Rendered to planner and critic as "Operator
    suggestions". A proposal source, never an authority: seeded material faces the critic
    unchanged, and coming from a handoff is not evidence.
  * the experiment store -- measured NEGATIVES, so the critic can refuse a re-derivation with
    a receipt rather than the planner spending a call rediscovering it. Recorded under a
    SYNTHETIC HISTORICAL EPOCH, so ``recall()`` marks them stale and the actors are told the
    mechanism transfers but the number does not.

That pairing is the point. Seed 02 ranks "async weight prefetch / double-buffering in the
GEMV" as a Tier-2 lever; ``akm-hist-q8-prefetch`` records that exact mechanism as measured
net-negative on gfx90a. The planner sees the lever, the critic sees the receipt, and the
rejection costs one planner call instead of a build and a GPU window.

Idempotent: ``archive.record`` is idempotent on attempt identity, and inbox files are
overwritten with identical content.
"""
from __future__ import annotations

import argparse
import hashlib
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

from . import archive

SEEDS = Path(__file__).resolve().parent / "seeds"


def install(store_root: Path, *, seeds: Path = SEEDS) -> dict:
    """Install hypothesis files and historical negatives into a store."""
    inbox = store_root / "inbox"
    inbox.mkdir(parents=True, exist_ok=True)
    installed = []
    for source in sorted((seeds / "hypotheses").glob("*.md")):
        shutil.copyfile(source, inbox / source.name)
        installed.append(source.name)

    body = json.loads((seeds / "negatives.json").read_text(encoding="utf-8"))
    epoch = archive.epoch_for(anchor_commit=body["epoch_anchor_commit"],
                              build_recipe=body["epoch_build_recipe"])
    now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    added = 0
    for record in body["records"]:
        # A seeded negative is a FIXED historical record, not an event in time: the
        # same seed re-run must not duplicate it. Attempt identity is otherwise
        # time-based (two transients minutes apart are genuinely distinct events), so
        # seeds carry an explicit content digest, which `_attempt_id` prefers.
        record = dict(record)
        record.setdefault("proposal_sha256", hashlib.sha256(json.dumps(
            record, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest())
        if archive.record(store_root, record, epoch=epoch, recorded_at=now,
                          campaign_id="ak-loop-seed"):
            added += 1
    return {"inbox_files": installed, "historical_epoch": epoch,
            "negatives_seen": len(body["records"]), "negatives_added": added}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--store", type=Path, required=True)
    args = parser.parse_args(argv)

    result = install(args.store)
    print(f"inbox      {len(result['inbox_files'])} file(s) -> {args.store / 'inbox'}")
    for name in result["inbox_files"]:
        print(f"             {name}")
    print(f"negatives  {result['negatives_added']} added "
          f"({result['negatives_seen']} in the seed file; the rest were already recorded)")
    print(f"           under historical epoch {result['historical_epoch'][:12]} "
          f"-- cross-epoch, so they render as STALE and their numbers are not comparable")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
