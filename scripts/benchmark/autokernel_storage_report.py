#!/usr/bin/env python3
"""Plan reclamation over the AutoKernel runtime tree. Reports; deletes only on --force.

WHY THIS EXISTS
---------------
`storage.expire_artifact()` is a complete, tombstone-first, journal-required
reclamation path with a dry-run default. It has **zero callers**. Meanwhile
`/mnt/raid0/llm/autokernel/` reached 41 GB across 62 deployments -- 17 GB of it build
directories, of which ~15.6 GB are duplicate anchor builds compiled from a
byte-identical tree because the build cache key hashes the CANDIDATE's patch (see
`discovery_static_registry._build_key_contract`). The disk filled while a working
reclaimer sat unused.

This is the missing caller. It is deliberately a REPORT by default:
`storage.expire_artifact` refuses without `force=True` and requires a journal on the
force path, because "journal unavailable, delete anyway" destroys the record the
design exists to keep (invariant 7). Nothing here widens that.

Only the three ratified expirable kinds are considered (§5.8): rejected candidate
build trees, retired campaign worktrees, stale profiler traces. A fourth would be a
ratification event, not a code change -- so an unrecognised directory is REPORTED AS
UNCLASSIFIED rather than swept up, because a reclaimer that guesses is how evidence
disappears.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kernel_rnd.autokernel import storage  # noqa: E402


def directory_bytes(path: Path) -> int:
    total = 0
    for item in path.rglob("*"):
        try:
            if item.is_file() and not item.is_symlink():
                total += item.stat().st_size
        except OSError:
            continue
    return total


def classify(path: Path) -> str | None:
    """Map a runtime directory to a RATIFIED expirable kind, or None.

    None means "this tool does not know what that is", and the caller reports it
    rather than reclaiming it.
    """
    parts = path.parts
    if "builds" in parts:
        return "rejected_candidate_build_tree"
    if "worktrees" in parts:
        return "retired_campaign_worktree"
    if path.name in {"rocprofv3", "diagnostics", "probes"} or "screens" in parts:
        return "stale_profiler_trace"
    return None


def survey(root: Path, *, depth: int = 2) -> list[dict]:
    rows = []
    for deployment in sorted(p for p in root.iterdir() if p.is_dir()):
        for child in sorted(p for p in deployment.iterdir() if p.is_dir()):
            kind = classify(child)
            rows.append({
                "path": str(child),
                "expirable_kind": kind,
                "bytes": directory_bytes(child),
            })
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path,
                        default=Path("/mnt/raid0/llm/autokernel/deployments"))
    parser.add_argument("--quota-gb", type=float, default=64.0,
                        help="campaign storage budget from the manifest (never invented "
                             "here; pass the manifest's budgets.max_storage_gb)")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--force", action="store_true",
                        help="ACTUALLY reclaim. Requires a journal; refuses without "
                             "one, because an unjournalled delete destroys the record "
                             "the retention design exists to keep.")
    args = parser.parse_args(argv)

    if not args.root.is_dir():
        print(f"REFUSED: no such runtime root: {args.root}", file=sys.stderr)
        return 2
    if args.force:
        print("REFUSED: --force needs an operator-supplied journal sink. This tool "
              "reports; reclamation authority is operator-only outside the narrow "
              "expirable classes (MEASUREMENT.md:223-229).", file=sys.stderr)
        return 3

    rows = survey(args.root)
    by_kind: dict[str, dict[str, int]] = {}
    for row in rows:
        key = row["expirable_kind"] or "UNCLASSIFIED"
        bucket = by_kind.setdefault(key, {"count": 0, "bytes": 0})
        bucket["count"] += 1
        bucket["bytes"] += row["bytes"]

    total = sum(row["bytes"] for row in rows)
    print(f"\nAutoKernel runtime survey: {args.root}")
    print(f"  directories: {len(rows)}    total: {total / 2**30:.2f} GB\n")
    print(f"  {'kind':<34} {'count':>6} {'GB':>10}")
    print(f"  {'-' * 34} {'-' * 6} {'-' * 10}")
    for kind in sorted(by_kind, key=lambda k: -by_kind[k]["bytes"]):
        bucket = by_kind[kind]
        print(f"  {kind:<34} {bucket['count']:>6} {bucket['bytes'] / 2**30:>10.2f}")

    reclaimable = sum(bucket["bytes"] for kind, bucket in by_kind.items()
                      if kind != "UNCLASSIFIED")
    print(f"\n  reclaimable under the three ratified expirable kinds: "
          f"{reclaimable / 2**30:.2f} GB")
    if "UNCLASSIFIED" in by_kind:
        print(f"  UNCLASSIFIED and therefore NOT reclaimable here: "
              f"{by_kind['UNCLASSIFIED']['bytes'] / 2**30:.2f} GB across "
              f"{by_kind['UNCLASSIFIED']['count']} directories -- reported, not swept, "
              f"because a reclaimer that guesses is how evidence disappears.")

    # The top offenders, since the point is to make the growth legible.
    print("\n  largest 10:")
    for row in sorted(rows, key=lambda r: -r["bytes"])[:10]:
        print(f"    {row['bytes'] / 2**30:>8.2f} GB  "
              f"{row['expirable_kind'] or 'UNCLASSIFIED':<32} {row['path']}")

    payload = {"schema": "epyc.autokernel.storage_survey.v1",
               "root": str(args.root), "total_bytes": total,
               "by_kind": by_kind, "rows": rows,
               "reclaimable_bytes": reclaimable,
               "expirable_kinds": sorted(storage.EXPIRABLE_KINDS)}
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "storage-survey.json").write_text(json.dumps(payload, indent=2),
                                                       encoding="utf-8")
        print(f"\nwrote {args.out / 'storage-survey.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
