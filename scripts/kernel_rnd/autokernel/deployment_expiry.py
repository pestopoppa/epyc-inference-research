#!/usr/bin/env python3
"""deployment_expiry.py - AKU-15: the first caller of `storage.expire_artifact`.

Reclaims `<deployments>/<dep>/builds/<build_key>` trees, tombstone-first, through the storage plane. Conservative by
construction; every gate below must pass or the build is KEPT with a reason:

  * DRY RUN unless `apply=True` (CLI `--apply`); `--apply` also needs a journal root. Nothing is "defaulted on".
  * FACTS: the build_key must have an entry in `<dep>/state/expiry-facts.json`
    {"<build_key>": {"expirable_kind": ..., "preconditions": {...}}} written by whoever retires the candidate/campaign.
    No entry => KEEP-NO-FACTS (an unreadable fact is not a true one; storage's own rule gates the rest).
  * TTL: newest mtime anywhere in the tree older than `ttl_days` (default 30).
  * not named by the deployment's state.json (in-flight/current), no held `controller.run.lock`/`supervisor.lock`.
  * SCOPE: paths under `/mnt/raid0/llm/kernels` are refused. That tree (`kernels/builds`, 7-day TTL) is owned by
    epyc-root `scripts/safety/build_ttl.py`; this module owns only `autokernel/deployments/*/builds`. The two never
    overlap, so no second competing expiry exists over the same paths.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

try:
    from . import storage as ST
except ImportError:  # run as a script
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
    from scripts.kernel_rnd.autokernel import storage as ST  # type: ignore

DEFAULT_ROOT = "/mnt/raid0/llm/autokernel/deployments"
KERNELS_STORE = "/mnt/raid0/llm/kernels"   # build_ttl.py's tree: never touched here
FACTS_NAME = "expiry-facts.json"
LIVE_LOCKS = ("controller.run.lock", "supervisor.lock")


def _locked(path: Path) -> bool:
    """True when another process holds the flock (or it cannot be probed: unknown counts as live)."""
    try:
        fd = os.open(path, os.O_RDONLY)
    except FileNotFoundError:
        return False
    except OSError:
        return True
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(fd, fcntl.LOCK_UN)
        return False
    except OSError:
        return True
    finally:
        os.close(fd)


def sweep(root: str = DEFAULT_ROOT, *, ttl_days: float = 30.0, apply: bool = False, journal: Any = None,
          now: Optional[float] = None, actor: str = "ak-deployment-expiry",
          max_reclaim: Optional[int] = None) -> list:
    """-> list of {deployment, build, verdict, reason, bytes}. Reclaims only when apply=True and a journal is given."""
    if apply and journal is None:
        raise ValueError("apply=True requires a journal: the tombstone is written before the bytes")
    now = time.time() if now is None else now
    rootp = Path(root).resolve()
    if ST._under(str(rootp), KERNELS_STORE) or ST._under(KERNELS_STORE, str(rootp)):
        raise ValueError(f"{root} overlaps {KERNELS_STORE}, which build_ttl.py owns")
    policy = ST.StoragePolicy(campaign_quota_gb=1e9, owned_roots=(str(rootp),))
    out, done = [], 0
    for dep in sorted(p for p in rootp.iterdir() if p.is_dir() and not p.is_symlink()):
        builds = dep / "builds"
        if not builds.is_dir() or builds.is_symlink():
            continue
        facts_path = dep / "state" / FACTS_NAME
        try:
            facts = json.loads(facts_path.read_text()) if facts_path.exists() else {}
        except (OSError, ValueError):
            facts = None
        live = any(_locked(dep / n) or _locked(dep / "state" / n) for n in LIVE_LOCKS)
        try:
            state_text = (dep / "state" / "state.json").read_text(errors="replace")
        except OSError:
            state_text = None
        for b in sorted(p for p in builds.iterdir() if p.is_dir() and not p.is_symlink()):
            rec = {"deployment": dep.name, "build": b.name, "verdict": "KEEP", "reason": "", "bytes": 0}
            out.append(rec)
            if facts is None:
                rec["reason"] = f"KEEP-FACTS-UNREADABLE {FACTS_NAME}"
                continue
            fact = facts.get(b.name) if isinstance(facts, Mapping) else None
            if not isinstance(fact, Mapping):
                rec["reason"] = "KEEP-NO-FACTS no expiry-facts entry for this build"
                continue
            if live:
                rec["reason"] = "KEEP-LIVE deployment lock held or unprobeable"
                continue
            if state_text is None or b.name in state_text:
                rec["reason"] = "KEEP-INFLIGHT named by state.json" if state_text else "KEEP-NO-STATE state.json unreadable"
                continue
            try:
                newest = max([b.lstat().st_mtime] + [os.lstat(os.path.join(d, f)).st_mtime
                                                     for d, _, fs in os.walk(b) for f in fs])
            except OSError as exc:
                rec["reason"] = f"KEEP-UNSTATABLE {exc}"
                continue
            if now - newest < ttl_days * 86400:
                rec["reason"] = f"KEEP-YOUNG {int((now - newest) / 86400)}d < {ttl_days}d"
                continue
            if max_reclaim is not None and done >= max_reclaim:
                rec["reason"] = f"KEEP-MAXCAP {max_reclaim}"
                continue
            try:
                art = ST.ExpirableArtifact(
                    path=str(b), campaign_id=dep.name, sha256=ST.hash_tree_manifest(b),
                    durability_class="hash_and_provenance_only", expirable_kind=str(fact.get("expirable_kind")),
                    reason=f"deployment build tree older than {ttl_days}d with retirement facts recorded",
                    rule_id="R-AKU-15-deployment-build-expiry", actor=actor, retention_class="expirable",
                    preconditions=dict(fact.get("preconditions") or {}))
                sink = ST.JournalTombstoneSink(journal) if apply else None
                res = ST.expire_artifact(art, policy, journal=sink, force=apply,
                                         now=datetime.fromtimestamp(now, timezone.utc))
            except (ST.StorageError, KeyError, TypeError, ValueError) as exc:
                rec["reason"] = f"KEEP-REFUSED {type(exc).__name__}: {exc}"[:300]
                continue
            rec.update(verdict="RECLAIMED" if res.deleted else "WOULD-RECLAIM", reason="", bytes=res.measured_size_bytes)
            done += 1
    return out


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--ttl-days", type=float, default=30.0)
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--journal-root", help="AK journal root; required with --apply")
    ap.add_argument("--max", type=int)
    a = ap.parse_args(argv)
    journal = None
    if a.apply:
        if not a.journal_root:
            ap.error("--apply requires --journal-root")
        from . import journal as J  # noqa: PLC0415
        journal = J.Journal(a.journal_root)
    recs = sweep(a.root, ttl_days=a.ttl_days, apply=a.apply, journal=journal, max_reclaim=a.max)
    for r in recs:
        print(f"AKU15 {r['verdict']} {r['deployment']}/{r['build'][:12]} bytes={r['bytes']} {r['reason']}")
    print(f"AKU15 SUMMARY builds={len(recs)} reclaim={sum(r['verdict'] in ('RECLAIMED', 'WOULD-RECLAIM') for r in recs)}"
          f" bytes={sum(r['bytes'] for r in recs)} mode={'apply' if a.apply else 'dry-run'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
