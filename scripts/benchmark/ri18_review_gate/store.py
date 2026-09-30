"""RI-18 run directory: per-stage records, the segment ledger, and the pinned run manifest.

Layout of ``--out``::

    run_manifest.json   pins (items, served API identity, code_root commit, snapshot, config)
    segments.jsonl      one line per segment event: start / end / void
    answers.jsonl       stage 1 (+ stage 2 quality mark)          one line per item
    verdicts.jsonl      stage 3 (stage=verdict) + stage 5 (stage=vfull)
    revisions.jsonl     stage 4 (stage=revise)
    noise.jsonl         noise controls (stage=noise_verdict | noise_revise)
    gate.jsonl          stage 6 (offline gate against the snapshot)

Records are appended with ``thesis_ufh13.records.append_record`` (fsync per line); a torn final
line is ignored, never repaired. Each record carries its ``segment_id``. A record counts only if
its segment is VALID: not voided, and -- for a GPU segment -- closed by an ``end`` event whose
canary pair passed in both directions at the start AND the end. A record of an invalid segment is
simply re-run on resume; the latest valid record per (stage, item) wins.

One process per stage at a time (``stage_lock``); different stages may run concurrently (the
GPU ``verdict`` segment can follow the CPU ``answer`` segment item by item).
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from ..thesis_ufh13.records import append_record

RECORD_SCHEMA = "ri18-review-gate-record/v1"
SEGMENT_SCHEMA = "ri18-review-gate-segment/v1"
MANIFEST_NAME = "run_manifest.json"
SEGMENTS_NAME = "segments.jsonl"
STAGE_FILES = {
    "answer": "answers.jsonl",
    "verdict": "verdicts.jsonl",
    "vfull": "verdicts.jsonl",
    "revise": "revisions.jsonl",
    "noise_verdict": "noise.jsonl",
    "noise_revise": "noise.jsonl",
    "gate": "gate.jsonl",
}
GPU_SEGMENTS = ("verdict", "noise-verdict")

# Fields that must match on resume (a changed one is a different experiment).
RESUME_KEYS = (
    "runner", "run_id", "dry_run", "items_sha256", "workload_manifest_sha256", "render",
    "split_sha256", "served", "code_root_commit", "snapshot", "generation", "body_keys",
    "user_id", "reviewer_role", "verdict_caps",
)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read_jsonl(path: Path, schema: str | None = None) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if schema is None or row.get("schema") == schema:
                rows.append(row)
    return rows


class RunDir:
    def __init__(self, out: Path) -> None:
        self.out = Path(out)

    # ── manifest ─────────────────────────────────────────────────────────
    @property
    def manifest_path(self) -> Path:
        return self.out / MANIFEST_NAME

    def open(self, manifest: dict[str, Any]) -> dict[str, Any]:
        """Create the manifest, or refuse to resume when a pinned field drifted."""
        self.out.mkdir(parents=True, exist_ok=True)
        if self.manifest_path.exists():
            old = json.loads(self.manifest_path.read_text())
            diffs = [k for k in RESUME_KEYS if old.get(k) != manifest.get(k)]
            if diffs:
                detail = {k: {"pinned": old.get(k), "now": manifest.get(k)} for k in diffs}
                raise ManifestDrift(f"refusing to resume {self.out}: manifest differs on {diffs}: "
                                    f"{json.dumps(detail, sort_keys=True)[:1500]}")
            return old
        doc = {**manifest, "created_at": now_iso()}
        tmp = self.manifest_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
        tmp.replace(self.manifest_path)
        return doc

    def manifest(self) -> dict[str, Any]:
        return json.loads(self.manifest_path.read_text())

    # ── segments ─────────────────────────────────────────────────────────
    def segment_event(self, event: dict[str, Any]) -> None:
        append_record(self.out / SEGMENTS_NAME,
                      {"schema": SEGMENT_SCHEMA, "at": now_iso(), **event})

    def new_segment(self, kind: str, info: dict[str, Any]) -> str:
        seg = f"{kind}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:6]}"
        self.segment_event({"event": "start", "segment_id": seg, "kind": kind, **info})
        return seg

    def segments(self) -> dict[str, dict[str, Any]]:
        """segment_id -> folded state {kind, started, ended, void, void_reasons, canaries_ok}."""
        out: dict[str, dict[str, Any]] = {}
        for ev in read_jsonl(self.out / SEGMENTS_NAME, SEGMENT_SCHEMA):
            seg = out.setdefault(ev["segment_id"], {"kind": ev.get("kind"), "started": None,
                                                     "ended": None, "void": False,
                                                     "void_reasons": [], "canaries_ok": None,
                                                     "exit": None})
            if ev.get("kind"):
                seg["kind"] = ev["kind"]
            if ev["event"] == "start":
                seg["started"] = ev.get("at")
            elif ev["event"] == "end":
                seg["ended"] = ev.get("at")
                seg["canaries_ok"] = ev.get("canaries_ok")
                seg["exit"] = ev.get("exit")
            elif ev["event"] == "void":
                seg["void"] = True
                seg["void_reasons"].append(ev.get("reason"))
        return out

    def valid_segment_ids(self, *, current: str | None = None) -> set[str]:
        valid = set()
        for seg_id, seg in self.segments().items():
            if seg["void"]:
                continue
            if seg["kind"] in GPU_SEGMENTS and seg_id != current:
                if not seg["ended"] or seg["canaries_ok"] is not True:
                    continue
            valid.add(seg_id)
        return valid

    # ── records ──────────────────────────────────────────────────────────
    def append(self, stage: str, record: dict[str, Any]) -> None:
        append_record(self.out / STAGE_FILES[stage],
                      {"schema": RECORD_SCHEMA, "stage": stage, "recorded_at": now_iso(), **record})

    def records(self, stage: str, *, current: str | None = None) -> dict[str, dict[str, Any]]:
        """item_id -> latest record of ``stage`` from a valid segment (``current`` counts)."""
        valid = self.valid_segment_ids(current=current)
        out: dict[str, dict[str, Any]] = {}
        for row in read_jsonl(self.out / STAGE_FILES[stage], RECORD_SCHEMA):
            if row.get("stage") == stage and row.get("segment_id") in valid:
                out[row["item_id"]] = row
        return out

    def all_records(self, stage: str) -> list[dict[str, Any]]:
        return [r for r in read_jsonl(self.out / STAGE_FILES[stage], RECORD_SCHEMA)
                if r.get("stage") == stage]

    @contextlib.contextmanager
    def stage_lock(self, stage: str) -> Iterator[None]:
        """One process per stage per run directory; a second one is refused, never queued."""
        self.out.mkdir(parents=True, exist_ok=True)
        path = self.out / f".lock.{stage}"
        handle = open(path, "a+")
        try:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise StageBusy(f"another process is running stage {stage!r} on {self.out}") from exc
            handle.seek(0)
            handle.truncate()
            handle.write(f"{os.getpid()}\n")
            handle.flush()
            yield
        finally:
            handle.close()


class ManifestDrift(RuntimeError):
    pass


class StageBusy(RuntimeError):
    pass
