#!/usr/bin/env python3
"""Ingest ONE completed v10 KV-quant sweep's exact sidecar into the Vidya ledger, idempotently.

    python3 scripts/benchmark/kv_quant_27b_v10_ingest.py <run-dir> [--dry-run] [--json]

Run by the sweep's owner after `kv_quant_27b_v10_sweep.py --execute` exits 0. It never runs a
sweep, never touches a GPU, and never backfills: a run directory without the producer's sidecar
and capture receipt (every pre-hook run, including the 2026-09-22 ones) is declined.

PRECONDITIONS (all, or nothing is written):

  * `summary.json` is the sweep summary schema with `status == "ok"`;
  * `belief_capture_receipt.json` is the producer's receipt: written, 12 rows, the capture
    schema and source kind, `grades_nothing`, naming THIS sidecar, and its `scored_sha256`
    equals the sha256 of this `summary.json`;
  * `belief_measurements.jsonl` holds exactly the 12 arm x depth x metric rows of ONE run named
    after the directory, each clean under the producer's own `validate_row()`, with unique
    measurement ids and the receipt's scored digest.

THE WRITE is ROOT's own: `ingest_sources.ingest(Ledger, "kv-quant-27b-v10-measurement", ...)`,
loaded from `EPYC_ROOT_REPO` (default /workspace) -- the same function `cli.py ingest` calls, so
the ROOT adapter projects and `claim_tuple.grade()` decides. Nothing here grades.

IDEMPOTENCY. ROOT's ingest appends every projected frame, so a second call would duplicate
evidence. This wrapper (a) holds an exclusive lock on the run directory, (b) pins `--as-of` to
the sidecar's own `emitted_at`, so a retry projects byte-identical frames, (c) checks the ledger
for the 12 claim ids first: all present exactly once -> `already_ingested`, nothing written;
some present -> `partial_ledger_state`, refused for a human; none -> dry run, verify the report
(1 unit, 12 rows, 36 frames, nothing refused/declined/missing), append, then re-verify the chain
and that each claim id now appears exactly once. The receipt `vidya_ingest_receipt.json` is
written beside the sidecar.

Exit codes: 0 ingested / already ingested / dry run clean; 2 refused (preconditions, report or
ledger verification); 3 declined (pre-hook or failed run: nothing to ingest).
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import importlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator, Mapping

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kv_quant_27b_v10_sweep as producer  # noqa: E402

SOURCE = producer.CAPTURE_SOURCE_KIND
SIDECAR = producer.CAPTURE_SIDECAR_NAME
SUMMARY = "summary.json"
CAPTURE_RECEIPT = "belief_capture_receipt.json"
INGEST_RECEIPT = "vidya_ingest_receipt.json"
INGEST_RECEIPT_SCHEMA = "epyc.vidya.kv_quant_27b_v10_ingest_receipt.v1"
LOCK_NAME = ".vidya-ingest.lock"
SUMMARY_SCHEMA = "epyc.kv_quant_27b_v10_sweep.summary.v1"
EXPECTED_KEYS = frozenset((cell.name, depth.name, metric) for cell in producer.CELLS
                          for depth in producer.DEPTHS for metric in producer.CAPTURE_METRICS)
EXPECTED_ROWS = len(EXPECTED_KEYS)             # 12
FRAMES_PER_ROW = 3                              # source + claim + supporting evidence
ROOT_REPO_ENV = "EPYC_ROOT_REPO"


class Refused(Exception):
    def __init__(self, status: str, reason: str, code: int = 2):
        super().__init__(reason)
        self.status, self.reason, self.code = status, reason, code


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise Refused("refused", f"{path.name} unreadable: {exc}") from exc


def check_run(run_dir: Path) -> list[dict[str, Any]]:
    """The 12 validated native rows, or `Refused` naming the first unmet precondition."""
    sidecar, summary_path = run_dir / SIDECAR, run_dir / SUMMARY
    receipt_path = run_dir / CAPTURE_RECEIPT
    if not sidecar.is_file() and not receipt_path.is_file():
        raise Refused("declined", f"no {SIDECAR} or {CAPTURE_RECEIPT}: a pre-hook or unfinished "
                                  "run; runs are never backfilled", 3)
    if not summary_path.is_file():
        raise Refused("refused", f"no {SUMMARY}")
    summary = _json(summary_path)
    if not isinstance(summary, dict) or summary.get("schema") != SUMMARY_SCHEMA:
        raise Refused("refused", f"{SUMMARY} is not {SUMMARY_SCHEMA}")
    if summary.get("status") != "ok":
        raise Refused("declined", f"sweep status is {summary.get('status')!r}, not 'ok': a failed "
                                  "matrix is never ingested", 3)
    receipt = _json(receipt_path) if receipt_path.is_file() else None
    if not isinstance(receipt, dict):
        raise Refused("refused", f"{CAPTURE_RECEIPT} missing or not an object")
    scored = _sha256(summary_path)
    problems = [
        name for name, ok in (
            ("written", receipt.get("written") is True),
            ("rows", receipt.get("rows") == EXPECTED_ROWS),
            ("schema", receipt.get("schema") == producer.CAPTURE_SCHEMA),
            ("source_kind", receipt.get("source_kind") == SOURCE),
            ("grades_nothing", receipt.get("grades_nothing") is True),
            ("path", Path(str(receipt.get("path") or "")).resolve() == sidecar.resolve()),
            ("scored_sha256", receipt.get("scored_sha256") == scored),
        ) if not ok]
    if problems:
        raise Refused("refused", f"{CAPTURE_RECEIPT} invalid: {', '.join(problems)}")
    try:
        rows = [json.loads(line) for line in sidecar.read_text(encoding="utf-8").splitlines()
                if line.strip()]
    except (OSError, ValueError) as exc:
        raise Refused("refused", f"{SIDECAR} unreadable: {exc}") from exc
    if len(rows) != EXPECTED_ROWS:
        raise Refused("refused", f"{SIDECAR} has {len(rows)} rows, not {EXPECTED_ROWS}")
    keys, ids = set(), set()
    for row in rows:
        issues = producer.validate_row(row)
        if issues:
            raise Refused("refused", "row refused by validate_row: " + "; ".join(issues))
        if row["run_id"] != run_dir.name:
            raise Refused("refused", f"row run_id {row['run_id']!r} is not the run directory")
        if row["scored_sha256"] != scored:
            raise Refused("refused", "row scored_sha256 differs from this summary.json")
        parts = str((row.get("extra") or {}).get("locator", "")).split(":")
        if len(parts) != 5 or parts[0] != "kvq" or parts[1] != row["run_id"]:
            raise Refused("refused", f"row locator {parts!r} does not bind this run")
        keys.add((parts[2], parts[3], parts[4]))
        ids.add(row["measurement_id"])
    if keys != EXPECTED_KEYS or len(ids) != EXPECTED_ROWS:
        raise Refused("refused", f"rows cover {len(keys)} of {EXPECTED_ROWS} arm x depth x metric "
                                 f"keys with {len(ids)} unique ids")
    return rows


def as_of_for(rows: list[Mapping[str, Any]]) -> str:
    """The sidecar's own emission time: deterministic, so a retry projects identical frames."""
    stamps = {str(row["emitted_at"]) for row in rows}
    if len(stamps) != 1:
        raise Refused("refused", f"rows carry {len(stamps)} different emitted_at stamps")
    stamp = stamps.pop()
    datetime.fromisoformat(stamp.replace("Z", "+00:00"))
    return stamp


@contextlib.contextmanager
def run_lock(run_dir: Path) -> Iterator[None]:
    handle = open(run_dir / LOCK_NAME, "a+")
    try:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise Refused("refused", "another ingest of this run holds the lock") from exc
        yield
    finally:
        handle.close()


@contextlib.contextmanager
def root_api(root: Path) -> Iterator[dict[str, Any]]:
    """ROOT's ledger + ingest dispatcher, from `<root>/scripts/vidya`, verified by file."""
    vidya = (root / "scripts" / "vidya").resolve()
    for name in ("ledger.py", "ingest_sources.py", "claim_tuple.py"):
        if not (vidya / name).is_file():
            raise Refused("unavailable", f"ROOT module missing: {vidya / name}")
    sys.path.insert(0, str(vidya))
    try:
        ledger = importlib.import_module("ledger")
        ingest_sources = importlib.import_module("ingest_sources")
        for module in (ledger, ingest_sources):
            if Path(module.__file__).resolve().parent != vidya:
                raise Refused("unavailable", f"{module.__name__} resolves outside ROOT {vidya}")
        if SOURCE not in ingest_sources.SOURCES:
            raise Refused("unavailable", f"ROOT ingest name {SOURCE!r} is not wired in {vidya}")
        yield {"Ledger": ledger.Ledger, "ingest": ingest_sources.ingest, "vidya": vidya}
    finally:
        with contextlib.suppress(ValueError):
            sys.path.remove(str(vidya))


def _claim_counts(ledger: Any, claim_ids: set[str]) -> tuple[dict[str, int], dict[str, int]]:
    claims = {cid: 0 for cid in claim_ids}
    support = {cid: 0 for cid in claim_ids}
    for record in ledger.read_all():
        frame = record.frame
        kind, assertion = str(frame.get("frame_type", "")), frame.get("assertion") or {}
        cid = assertion.get("claim_id")
        if cid in claims and kind.endswith("claim_proposed/v1"):
            claims[cid] += 1
        elif cid in support and kind.endswith("evidence_supports_claim/v1"):
            support[cid] += 1
    return claims, support


def _check_report(report: Mapping[str, Any]) -> None:
    expected = {"units_matched": 1, "units_projected": 1, "rows_projected": EXPECTED_ROWS,
                "frames_emitted": EXPECTED_ROWS * FRAMES_PER_ROW}
    wrong = {k: report.get(k) for k, v in expected.items() if report.get(k) != v}
    for key in ("refused", "declined", "missing"):
        if report.get(key):
            wrong[key] = report[key]
    if wrong:
        raise Refused("refused", f"ingest report differs from one complete run: {wrong}")


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def ingest_run(run_dir: str | Path, *, root: str | Path | None = None,
               ledger_path: str | Path | None = None, dry_run: bool = False) -> dict[str, Any]:
    """One complete run into the ledger. Returns the receipt; raises only `Refused`."""
    run_dir = Path(run_dir).resolve()
    root = Path(root or os.environ.get(ROOT_REPO_ENV) or "/workspace")
    ledger_path = Path(ledger_path) if ledger_path else root / ".vidya" / "ledger.jsonl"
    rows = check_run(run_dir)
    as_of = as_of_for(rows)
    claim_ids = {f"clm_{row['measurement_id']}" for row in rows}
    with run_lock(run_dir), root_api(root) as api:
        ledger = api["Ledger"](ledger_path)
        if ledger_path.exists():
            errors = ledger.verify()
            if errors:
                raise Refused("refused", f"ledger integrity failed before ingest: {errors[0]}")
        claims, support = (_claim_counts(ledger, claim_ids) if ledger_path.exists()
                           else ({c: 0 for c in claim_ids}, {c: 0 for c in claim_ids}))
        present = {cid for cid, n in claims.items() if n}
        base = {"schema": INGEST_RECEIPT_SCHEMA, "run": run_dir.name, "run_dir": str(run_dir),
                "source_kind": SOURCE, "as_of": as_of, "ledger": str(ledger_path),
                "root": str(api["vidya"]), "claim_ids": sorted(claim_ids),
                "sidecar_sha256": _sha256(run_dir / SIDECAR), "dry_run": dry_run}
        if present == claim_ids and all(claims[c] == 1 and support[c] == 1 for c in claim_ids):
            return {**base, "status": "already_ingested", "frontier_before": len(ledger),
                    "frontier_after": len(ledger), "frames_appended": 0}
        if present:
            raise Refused("refused", f"partial_ledger_state: {len(present)} of {EXPECTED_ROWS} "
                                     "claim ids already in the ledger (or duplicated); a human "
                                     "must reconcile before any append")
        report = api["ingest"](ledger, SOURCE, [run_dir], as_of=as_of, dry_run=True)
        _check_report(report)
        if dry_run:
            return {**base, "status": "dry_run_clean", "report": report,
                    "frontier_before": len(ledger) if ledger_path.exists() else 0}
        before = len(ledger) if ledger_path.exists() else 0
        report = api["ingest"](ledger, SOURCE, [run_dir], as_of=as_of, dry_run=False)
        _check_report(report)
        errors = ledger.verify()
        after = len(ledger)
        claims, support = _claim_counts(ledger, claim_ids)
        if errors or after - before != EXPECTED_ROWS * FRAMES_PER_ROW or any(
                claims[c] != 1 or support[c] != 1 for c in claim_ids):
            raise Refused("refused", f"ledger verification failed after ingest: errors={errors[:1]} "
                                     f"appended={after - before} claims={sorted(set(claims.values()))}")
        receipt = {**base, "status": "ingested", "report": report, "frontier_before": before,
                   "frontier_after": after, "frames_appended": after - before}
        _write_json(run_dir / INGEST_RECEIPT, receipt)
        return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--root", type=Path, default=None,
                        help="ROOT checkout (default EPYC_ROOT_REPO or /workspace)")
    parser.add_argument("--ledger", type=Path, default=None,
                        help="ledger path (default <root>/.vidya/ledger.jsonl)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    try:
        result, code = ingest_run(args.run_dir, root=args.root, ledger_path=args.ledger,
                                  dry_run=args.dry_run), 0
    except Refused as exc:
        result, code = {"status": exc.status, "reason": exc.reason, "run_dir": str(args.run_dir)}, exc.code
    if args.json:
        print(json.dumps(result, sort_keys=True))
    else:
        print(f"{result['status']}: {result.get('reason') or result.get('run')}"
              + (f" (frames appended {result['frames_appended']})"
                 if "frames_appended" in result else ""))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
