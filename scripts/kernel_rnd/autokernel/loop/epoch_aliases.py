#!/usr/bin/env python3
"""Full-epoch -> measurement-epoch aliases for the AutoKernel loop (OP-60).

WHY
---
The loop's full epoch hashes the anchor commit, the build recipe and the declared host
state, and the host state carries `enrolled_manifest_digest`, which folds the
campaign's ACTOR roster in. DS41 2026-09-26 switched the critic model and the full
epoch moved e0aefe6a -> e384c2ad (half screen) and 4e841d83 -> 4a886e98 (unscreened)
with the anchor, target, recipe, instrument, requests and floor unchanged. Resume was
moved onto the measurement epoch first (d643d794); OP-60 (operator, 2026-09-26) moves
planner-history comparability and the P-AK-SEARCH-1-A3 do-not-repeat gate too.

Archive rows keep the FULL epoch as their provenance key and carry no measurement
epoch of their own. The mapping lives in the store's `epoch_aliases` table, one
self-verifying record per full epoch (`experiments.epoch_alias_record`): it holds the
anchor, recipe, full host state and measurement digest, and every reader recomputes
both digests before trusting it. A full epoch with no verified alias stays on
full-epoch comparison -- comparability never widens to an unknown measurement identity.

WHO WRITES ALIASES
------------------
* every launch registers its own (full, measurement) pair at start
  (`register_launch_alias`, from `run.main`);
* launches that predate OP-60 are backfilled from their recorded `loop-run.json`
  (`backfill`, this module's CLI): the full epoch is RE-DERIVED from the launch's
  recorded identity (anchor, argv, target, screen state, execution and request
  digests) plus the resolved campaign whose manifest digest it enrolled, and a record
  is admitted only when the re-derivation reproduces the full epoch the launch
  recorded. Dry run by default; `--apply` writes one idempotent record per epoch.

    PYTHONPATH=scripts/kernel_rnd python3 -m autokernel.loop.epoch_aliases \\
        --campaign-dir /mnt/raid0/llm/autokernel/campaigns/<id> [--apply]
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence

from ..controller import experiments

REPORT_SCHEMA = "epyc.autokernel.epoch_alias_backfill.v1"
MATCHED_INSTRUMENT = "matched_process_v2"      # serving.MATCHED_INSTRUMENT
DEFAULT_SERVING_PAIRS = 5                       # run.py --serving-pairs default


def _canonical_digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def launch_epoch_inputs(*, cpu_execution_digest: str | None = None,
                        gpu_execution_digest: str | None = None,
                        frozen_prompt_digest: str | None = None,
                        enrolled_manifest_digest: str | None = None,
                        enrolled_target: Mapping[str, Any] | None = None,
                        screen_state: Mapping[str, Any] | None = None,
                        serving_instrument: Mapping[str, Any] | None = None
                        ) -> dict[str, Any]:
    """The full epoch's host state, as `run.main` declares it. ONE derivation: the
    launch and the backfill both call this, so they cannot drift apart."""
    inputs: dict[str, Any] = {}
    if cpu_execution_digest is not None:
        inputs.update(cpu_execution_digest=cpu_execution_digest,
                      frozen_prompt_digest=frozen_prompt_digest)
    if gpu_execution_digest is not None:
        inputs.update(gpu_execution_digest=gpu_execution_digest,
                      frozen_prompt_digest=frozen_prompt_digest)
    if enrolled_manifest_digest is not None:
        if enrolled_target is None:
            raise ValueError("an enrolled manifest needs its selected target")
        inputs.update(enrolled_manifest_digest=enrolled_manifest_digest,
                      enrolled_target_digest=_canonical_digest(dict(enrolled_target)))
    if screen_state is not None:
        inputs["cpu_screen"] = dict(screen_state)
    if serving_instrument is not None:
        inputs["serving_instrument"] = dict(serving_instrument)
    return inputs


def register_launch_alias(store_root: Path, *, anchor_commit: str,
                          build_recipe: Mapping[str, Any],
                          epoch_inputs: Mapping[str, Any], measurement_digest: str,
                          source: Mapping[str, Any], recorded_at: str) -> str:
    """Record this launch's own mapping; `added` | `present` | `conflict`."""
    record = experiments.epoch_alias_record(
        anchor_commit=anchor_commit, build_recipe=build_recipe, host_state=epoch_inputs,
        measurement_digest=measurement_digest, source=source)
    with experiments.ExperimentStore(store_root) as store:
        return store.register_epoch_alias(record, recorded_at=recorded_at)


# ------------------------------------------------------------------ backfill

def _flag(argv: Sequence[str], name: str) -> str | None:
    for index, item in enumerate(argv):
        if item == name and index + 1 < len(argv):
            return argv[index + 1]
        if item.startswith(name + "="):
            return item.split("=", 1)[1]
    return None


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _prompt_digest(path: str | None) -> str | None:
    if not path:
        return None
    try:
        body = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    digest = body.get("digest") if isinstance(body, dict) else None
    return digest if isinstance(digest, str) else None


def load_resolved(paths: Iterable[Path]) -> tuple[list[tuple[Path, Any]], list[dict]]:
    """Resolved campaigns by path; unreadable ones are reported, never guessed."""
    from . import campaign_cli
    loaded, errors = [], []
    for path in paths:
        try:
            loaded.append((Path(path), campaign_cli.load_previous(Path(path))))
        except Exception as exc:     # noqa: BLE001 -- reported per path
            errors.append({"path": str(path), "reason": f"{type(exc).__name__}: {exc}"[:300]})
    return loaded, errors


def derive_from_loop_run(path: Path, resolved: Sequence[tuple[Path, Any]]
                         ) -> tuple[dict | None, str]:
    """(alias record, "") or (None, reason). Admitted only when the re-derived full
    epoch equals the one the launch recorded (and its measurement epoch, if any)."""
    from ..controller import build_recipe
    try:
        body = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return None, f"unreadable: {exc}"
    recorded = body.get("epoch")
    anchor = body.get("anchor_commit")
    target = body.get("target")
    if not isinstance(recorded, str) or not isinstance(anchor, str):
        return None, "no recorded epoch/anchor"
    if not isinstance(target, Mapping) or not target.get("manifest_digest"):
        return None, "no enrolled manifest: the full epoch IS the measurement epoch"
    original_target = target.get("original_target")
    if not isinstance(original_target, Mapping):
        return None, "no recorded original_target"
    manifest_digest = target["manifest_digest"]
    match = next(((p, r) for p, r in resolved if r.manifest_digest == manifest_digest), None)
    if match is None:
        return None, f"no resolved campaign with manifest digest {manifest_digest[:12]}"
    resolved_path, resolved_campaign = match
    continuation = body.get("continuation") or {}
    argv = list(((continuation.get("binding") or {}).get("argv")) or [])
    if not argv:
        return None, "no recorded launch argv"
    profile = continuation.get("cpu_profile_reference") or {}
    screen = body.get("cpu_screen")
    cpu = _flag(argv, "--cpu-serving-launch") is not None
    gpu = _flag(argv, "--gpu-serving-launch") is not None
    if gpu and not cpu:
        # A GPU launch records no execution digest in loop-run.json.
        return None, "GPU launch: execution digest not recorded"
    if not cpu:
        return None, "no serving launch in argv"
    execution = (screen.get("measured_execution_digest") if isinstance(screen, Mapping)
                 else profile.get("execution_digest"))
    prompts = (profile.get("prompt_manifest_digest")
               or _prompt_digest(_flag(argv, "--frozen-prompts")))
    if not execution or not prompts:
        return None, "execution or request digest not recorded"
    instrument = _flag(argv, "--serving-instrument")
    pairs = int(_flag(argv, "--serving-pairs") or DEFAULT_SERVING_PAIRS)
    inputs = launch_epoch_inputs(
        cpu_execution_digest=execution, frozen_prompt_digest=prompts,
        enrolled_manifest_digest=manifest_digest, enrolled_target=original_target,
        screen_state=screen if isinstance(screen, Mapping) else None,
        serving_instrument=({"version": instrument, "pairs": pairs}
                            if instrument == MATCHED_INSTRUMENT else None))
    record = experiments.epoch_alias_record(
        anchor_commit=anchor, build_recipe=build_recipe.NATIVE_CPU_RECIPE.to_dict(),
        host_state=inputs, measurement_digest=resolved_campaign.measurement_digest,
        source={"kind": "loop_run_backfill", "loop_run": str(path),
                "loop_run_sha256": _file_sha256(Path(path)),
                "resolved_campaign": str(resolved_path),
                "resolved_campaign_sha256": _file_sha256(resolved_path)})
    if record["full_epoch_sha256"] != recorded:
        return None, (f"re-derived full epoch {record['full_epoch_sha256'][:12]} != "
                      f"recorded {recorded[:12]}")
    if body.get("measurement_epoch") not in (None, record["measurement_epoch_sha256"]):
        return None, "re-derived measurement epoch differs from the recorded one"
    return record, ""


def backfill(store_root: Path, *, loop_runs: Sequence[Path],
             resolved_paths: Sequence[Path], apply: bool = False,
             recorded_at: str | None = None) -> dict[str, Any]:
    """Derive one alias per full epoch found in the store; write only with `apply`."""
    import sqlite3
    resolved, resolved_errors = load_resolved(resolved_paths)
    derived: dict[str, dict] = {}
    reasons: dict[str, list[str]] = {}
    for path in loop_runs:
        try:
            recorded = json.loads(Path(path).read_text(encoding="utf-8")).get("epoch")
        except (OSError, json.JSONDecodeError, AttributeError):
            recorded = None
        if not isinstance(recorded, str) or recorded in derived:
            continue
        record, reason = derive_from_loop_run(Path(path), resolved)
        if record is not None:
            derived[recorded] = record
        else:
            reasons.setdefault(recorded, []).append(f"{path}: {reason}")
    # Unbounded read-only open: this offline command re-verifies records between
    # queries, which the 0.2 s shared-history bound is not sized for.
    with experiments.ExperimentStore(store_root, read_only=not apply,
                                     bounded=False) as store:
        existing = store.epoch_aliases()
        counts = dict(store._connection.execute(
            "SELECT epoch_sha256, COUNT(*) FROM experiments GROUP BY epoch_sha256"
        ).fetchall())
        epochs = []
        for full, rows in sorted(counts.items(), key=lambda item: item[0]):
            entry: dict[str, Any] = {"full_epoch_sha256": full, "rows": rows}
            if full in existing:
                entry.update(action="present", measurement_epoch_sha256=existing[full])
            elif full in derived:
                record = derived[full]
                entry.update(measurement_epoch_sha256=record["measurement_epoch_sha256"],
                             source=record["source"]["loop_run"])
                if apply:
                    try:
                        entry["action"] = store.register_epoch_alias(
                            record, recorded_at=recorded_at or _now())
                    except (ValueError, sqlite3.DatabaseError) as exc:
                        entry.update(action="refused", reason=str(exc))
                else:
                    entry["action"] = "would_add"
            else:
                entry.update(action="unresolved", measurement_epoch_sha256=None,
                             reason=("; ".join(reasons.get(full, []))[:600]
                                     or "no launch record names this epoch"))
            epochs.append(entry)
    return {"schema": REPORT_SCHEMA, "store": str(store_root), "applied": bool(apply),
            "loop_runs_examined": len(loop_runs),
            "resolved_campaigns": [str(path) for path, _ in resolved],
            "resolved_errors": resolved_errors, "epochs": epochs,
            "rows_resolvable": sum(e["rows"] for e in epochs
                                   if e.get("measurement_epoch_sha256")),
            "rows_unresolved": sum(e["rows"] for e in epochs
                                   if not e.get("measurement_epoch_sha256"))}


def _now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python3 -m autokernel.loop.epoch_aliases",
        description="epoch-aliases: backfill full->measurement epoch aliases (OP-60). "
                    "Dry run unless --apply.")
    parser.add_argument("--campaign-dir", type=Path,
                        help="campaign root: store/, state*/batches/*/loop-run.json, "
                             "inputs/**/campaign-resolved.json*")
    parser.add_argument("--store", type=Path)
    parser.add_argument("--loop-run", type=Path, action="append", default=[])
    parser.add_argument("--resolved", type=Path, action="append", default=[])
    parser.add_argument("--apply", action="store_true",
                        help="write one idempotent alias record per derivable epoch")
    args = parser.parse_args(argv)
    store = args.store
    loop_runs = list(args.loop_run)
    resolved = list(args.resolved)
    if args.campaign_dir is not None:
        root = args.campaign_dir
        store = store or root / "store"
        loop_runs += sorted(root.glob("state*/batches/*/loop-run.json"))
        resolved += sorted(p for p in root.glob("inputs/**/campaign-resolved.json*")
                           if p.is_file())
    if store is None:
        parser.error("--store or --campaign-dir is required")
    try:
        report = backfill(store, loop_runs=loop_runs, resolved_paths=resolved,
                          apply=args.apply)
    except (OSError, ValueError) as exc:
        print(f"epoch-aliases: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


__all__ = ["backfill", "derive_from_loop_run", "launch_epoch_inputs", "load_resolved",
           "main", "register_launch_alias"]


if __name__ == "__main__":
    raise SystemExit(main())
