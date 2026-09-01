#!/usr/bin/env python3
"""The loop's live status, written for the dashboard to read.

WHY THIS EXISTS
---------------
The operator's first question about the rebuilt loop was "I have zero visibility on
what's going on" -- and they were right: the run was a process the dashboard knew
nothing about, while the dashboard correctly reported the SUPERSEDED deployment as
stopped. A loop nobody can see is a loop nobody can trust.

The data contract lives here, with the subsystem it observes; the page, the nav entry
and the health probe live with the hub (`dashboard/README.md`, the plane rule).

THREE THINGS THIS CARRIES THAT THE OLD SURFACE DID NOT

  * **A freshness envelope.** `generated_at` plus `stale_after_s`, so a reader can
    tell "still true" from "last true an hour ago". The superseded funnel could show
    a clean, empty, trusted page over a dead producer.
  * **GPU utilization.** Held seconds against busy seconds. The loop ran 95.4% idle
    on a held device for a month and nothing reported it, because the surface
    reported iterations and receipts.
  * **The negatives.** Every disposition, not just the keeps. A board that shows only
    wins is how 0 promotions looked like progress.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

STATUS_SCHEMA = "epyc.autokernel.loop_status.v1"
STATUS_FILENAME = "loop-status.json"
#: An iteration is minutes, dominated by the planner call. Past this the reading is
#: not "the loop is quiet", it is "nobody has heard from the loop".
DEFAULT_STALE_AFTER_S = 1800


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def write_json(store_root: Path, name: str, body: Any, *,
               prefix: str = ".status-") -> Path:
    """Publish one JSON document into the store, atomically.

    Extracted from `write` because a second publisher (`production.py`, the
    champion-vs-production bundle the dashboard headline reads) needs exactly this
    and a second copy of a durability primitive is a second thing to get subtly
    wrong. `prefix` names the scratch file so a store holding several publishers'
    temporaries stays greppable -- and so a test that asserts THIS publisher left
    none behind is still asserting over a pattern this publisher actually uses.
    """
    store_root.mkdir(parents=True, exist_ok=True)
    target = store_root / name
    handle, temporary = tempfile.mkstemp(dir=str(store_root), prefix=prefix)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(body, stream, indent=2, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
    return target


def write(store_root: Path, *, state: str, epoch: str, campaign_id: str,
          anchor_commit: str, surface: str, pairs: int,
          noise_floor_pct: float | None,
          model: str | None = None,
          outcomes: Sequence[Mapping[str, Any]] = (),
          iterations_planned: int = 0,
          champion_head: str | None = None,
          gpu: Mapping[str, Any] | None = None,
          hotspots: Sequence[Mapping[str, Any]] = (),
          step: str | None = None,
          anchor_guard: Mapping[str, Any] | None = None,
          stale_after_s: int = DEFAULT_STALE_AFTER_S) -> Path:
    """Atomically publish the loop's current standing.

    Atomic because a dashboard polling a half-written file is how a surface reports
    something that was never true. `state` is what the loop is DOING, and it is
    written at start, after every iteration, and on exit -- including on failure, so
    a crashed loop says `failed` rather than going quiet and looking merely slow.
    """
    counts: dict[str, int] = {}
    for row in outcomes:
        status = str(row.get("status", "unknown"))
        counts[status] = counts.get(status, 0) + 1

    measured = sum(counts.get(key, 0)
                   for key in ("kept", "measured_null", "keep_candidate"))
    body = {
        "schema": STATUS_SCHEMA,
        "generated_at": _now(),
        "stale_after_s": int(stale_after_s),
        "state": state,
        # The rung this run measures on (§5.3): with two rungs live, run histories
        # that do not carry their model would silently merge across instruments.
        "model": model,
        # What the loop is doing RIGHT NOW. An iteration can run far longer
        # than the freshness envelope -- a single planner call exceeded 18
        # minutes once the bundle carried the program and the seeds -- so
        # without a sub-iteration beat a healthy loop reads as stale.
        "step": step,
        "campaign_id": campaign_id,
        "epoch_sha256": epoch,
        "anchor_commit": anchor_commit,
        "surface": surface,
        "pairs": pairs,
        # The bar a candidate must clear, and where it came from. A loop whose
        # threshold sits below its own instrument's resolution is the defect this
        # rebuild exists to close, so the number is on the surface.
        "noise_floor_pct": noise_floor_pct,
        "iterations_planned": int(iterations_planned),
        "iterations_done": len(outcomes),
        "measurements_reached": measured,
        "dispositions": dict(sorted(counts.items())),
        "champion_head": champion_head,
        # The last promotion A/A: did the binary in the anchor slot prove to BE the
        # champion. `null` means no promotion has happened on this run, which is a
        # different fact from "the check passed" and must stay distinguishable.
        "anchor_guard": dict(anchor_guard) if anchor_guard else None,
        "gpu": dict(gpu or {}),
        "hotspots": [dict(row) for row in hotspots][:12],
        # Newest first: the operator reads the top of the list.
        "recent": [
            {"status": row.get("status"),
             "mechanism_id": row.get("mechanism_id"),
             "effect_fraction": row.get("effect_fraction"),
             "reason": (row.get("reason") or "")[:240]}
            for row in list(outcomes)[-10:][::-1]],
    }

    return write_json(store_root, STATUS_FILENAME, body)


def read(store_root: Path) -> dict[str, Any] | None:
    path = Path(store_root) / STATUS_FILENAME
    if not path.is_file():
        return None
    try:
        body = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return body if isinstance(body, dict) else None


def freshness(body: Mapping[str, Any] | None, *, now: datetime | None = None
              ) -> dict[str, Any]:
    """Three-valued, never two.

    `absent` is not `stale` is not `fresh`. Collapsing absent into stale is how a
    dead producer renders as a clean empty page -- the exact defect recorded in
    `dashboard/server.py`'s own comment about `[]`-vs-`null`.
    """
    if body is None:
        return {"state": "absent", "age_s": None,
                "detail": "no loop-status.json; the loop has never run here"}
    stamped = body.get("generated_at")
    try:
        written = datetime.fromisoformat(str(stamped).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return {"state": "malformed", "age_s": None,
                "detail": f"unparseable generated_at {stamped!r}"}
    age = ((now or datetime.now(timezone.utc)) - written).total_seconds()
    limit = float(body.get("stale_after_s") or DEFAULT_STALE_AFTER_S)
    return {
        "state": "fresh" if age <= limit else "stale",
        "age_s": round(age, 1),
        "stale_after_s": limit,
        "detail": (f"last heard from the loop {age / 60:.1f} min ago"
                   if age > limit else "current"),
    }


__all__ = ["DEFAULT_STALE_AFTER_S", "STATUS_FILENAME", "STATUS_SCHEMA", "freshness",
           "read", "write", "write_json"]
