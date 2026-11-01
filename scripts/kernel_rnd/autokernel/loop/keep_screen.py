"""Versioned, fail-closed publisher for the AKX per-keep dashboard projection.

Observed screens are immutable records bound to a keep, resolved campaign and
workload signatures.  The dashboard projection is rebuilt on every status
heartbeat; a missing screen is represented there as UNKNOWN and is never
persisted as an immutable observation.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import tempfile
from typing import Any, Mapping, Sequence

STATUS_SCHEMA = "epyc.autokernel.keep_screen_status.v1"
RECORD_SCHEMA = "epyc.autokernel.keep_screen_record.v1"
ROW_SCHEMA = "epyc.autokernel.blast_radius_row.v1"
SCREEN_SCHEMA = "epyc.autokernel.blast_radius_screen.v1"
REFUSAL_SCHEMA = "epyc.autokernel.nonregression_refusal.v1"
RECORD_DIR = "keep-screens"
_KEEP_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,159}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CLASSES = {"INERT", "TOUCHED", "UNKNOWN"}
_VERDICTS = {"REGRESSION", "NOT_SHOWN_NON_INFERIOR"}
MAX_KEEPS = 128
MAX_WORKLOADS = 64
MAX_RECORD_BYTES = 1024 * 1024
MAX_SOURCE_BYTES = 256 * 1024 * 1024
FUTURE_SKEW_TOLERANCE_S = 300.0


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def record_path(store_root: Path | str, keep_id: str) -> Path:
    if not isinstance(keep_id, str) or not _KEEP_ID.fullmatch(keep_id):
        raise ValueError("keep_id is not a safe bounded identifier")
    return Path(store_root) / RECORD_DIR / f"{_digest(keep_id)}.json"


def _read_record(path: Path) -> dict:
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC)
    with os.fdopen(fd, "rb") as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode) or not 0 < before.st_size <= MAX_RECORD_BYTES:
            raise ValueError("screen record is not a bounded regular file")
        raw = stream.read(MAX_RECORD_BYTES + 1)
        after = os.fstat(stream.fileno())
    identity = lambda info: (info.st_dev, info.st_ino, info.st_mtime_ns, info.st_size)
    if len(raw) != before.st_size or identity(before) != identity(after):
        raise ValueError("screen record changed while being read")
    body = json.loads(raw)
    if not isinstance(body, dict):
        raise ValueError("screen record is not a JSON object")
    return body


def _valid_row(row: Mapping[str, Any], *, keep_id: str, workload_id: str) -> bool:
    if (row.get("schema") != ROW_SCHEMA or row.get("keep_id") != keep_id
            or row.get("workload_id") != workload_id
            or row.get("classification") not in _CLASSES):
        return False
    classification, tier = row["classification"], row.get("tier")
    if classification == "INERT":
        if row.get("decision_class") != "INERT" or tier not in {"T0", "T1"}:
            return False
        witnesses = row.get("witnesses")
        if not isinstance(witnesses, Mapping) or not witnesses:
            return False
        if tier == "T0":
            anchors, candidates = (witnesses.get("dso_digests_anchor"),
                                   witnesses.get("dso_digests_candidate"))
            return (isinstance(anchors, Mapping) and bool(anchors) and anchors == candidates
                    and all(isinstance(name, str) and isinstance(digest, str)
                            and _SHA256.fullmatch(digest)
                            for name, digest in anchors.items()))
        profiles = witnesses.get("coverage_profile_sha256")
        if not isinstance(profiles, list) or len(profiles) != 2:
            return False
        hashes = [*profiles, witnesses.get("shape_envelope_hash")]
        return all(isinstance(item, str) and _SHA256.fullmatch(item) for item in hashes)
    if classification == "TOUCHED":
        return row.get("decision_class") == "TOUCHED" and row.get("tier") is None
    return (row.get("decision_class") == "TOUCHED" and row.get("tier") is None
            and isinstance(row.get("unknown_reason"), str)
            and bool(row["unknown_reason"]))


def _validate_record(record: Any, *, keep_id: str,
                     workloads: Mapping[str, str], campaign_id: str,
                     manifest_digest: str) -> dict | None:
    if not isinstance(record, dict) or record.get("schema") != RECORD_SCHEMA:
        return None
    fields = {"schema", "keep_id", "campaign_id", "campaign_manifest_sha256",
              "workload_signatures",
              "source_path", "source_sha256", "observed_at", "screen", "rows",
              "refusals", "record_sha256"}
    if set(record) != fields or record.get("keep_id") != keep_id \
            or record.get("campaign_id") != campaign_id \
            or record.get("campaign_manifest_sha256") != manifest_digest:
        return None
    signatures = record.get("workload_signatures")
    if (not isinstance(signatures, dict) or set(signatures) != set(workloads)
            or signatures != dict(workloads)):
        return None
    if (not isinstance(record.get("source_path"), str)
            or not Path(record["source_path"]).is_absolute()
            or not isinstance(record.get("source_sha256"), str)
            or not _SHA256.fullmatch(record["source_sha256"])):
        return None
    unsigned = dict(record)
    digest = unsigned.pop("record_sha256", None)
    if digest != _digest(unsigned):
        return None
    screen, rows, refusals = record.get("screen"), record.get("rows"), record.get("refusals")
    if (not isinstance(screen, dict) or screen.get("schema") != SCREEN_SCHEMA
            or screen.get("keep_id") != keep_id
            or not isinstance(rows, list) or len(rows) != len(workloads)
            or not isinstance(refusals, list)):
        return None
    expected_ids = set(workloads)
    found: dict[str, dict] = {}
    for row in rows:
        if not isinstance(row, dict):
            return None
        workload_id = row.get("workload_id")
        if workload_id not in expected_ids or workload_id in found \
                or not _valid_row(row, keep_id=keep_id, workload_id=workload_id):
            return None
        found[workload_id] = row
    if set(found) != expected_ids or screen.get("workload_ids") != sorted(expected_ids):
        return None
    refs = [_digest(found[key]) for key in sorted(found)]
    if screen.get("rows_ref") != refs:
        return None
    for name, value in (("touched_ids", "TOUCHED"), ("inert_ids", "INERT"),
                        ("unknown_ids", "UNKNOWN")):
        if screen.get(name) != sorted(k for k, row in found.items()
                                     if row["classification"] == value):
            return None
    declared_screen_digest = screen.get("screen_digest")
    if not isinstance(declared_screen_digest, str) or not _SHA256.fullmatch(declared_screen_digest):
        return None
    refusal_workloads = set()
    for refusal in refusals:
        if (not isinstance(refusal, dict) or refusal.get("schema") != REFUSAL_SCHEMA
                or refusal.get("keep_id") != keep_id
                or refusal.get("workload_id") not in expected_ids
                or refusal.get("verdict") not in _VERDICTS
                or not isinstance(refusal.get("remedy"), dict)
                or type(refusal["remedy"].get("suggested_form")) is not int
                or refusal["remedy"].get("suggested_form") not in {1, 2, 3}
                or not isinstance(refusal["remedy"].get("discriminator_hint"), str)
                or refusal.get("blast_radius_row_ref") !=
                   _digest(found[refusal["workload_id"]])
                or refusal["workload_id"] in refusal_workloads):
            return None
        refusal_workloads.add(refusal["workload_id"])
    return {"screen": screen, "rows": found, "refusals": refusals,
            "record_sha256": digest, "source_path": record["source_path"],
            "source_sha256": record["source_sha256"], "observed_at": record["observed_at"]}


def write_record(store_root: Path | str, *, keep_id: str, campaign_id: str,
                 campaign_manifest_sha256: str, source_path: str,
                 source_sha256: str, workload_signatures: Mapping[str, str],
                 observed_at: str, screen: Mapping[str, Any],
                 rows: Sequence[Mapping[str, Any]],
                 refusals: Sequence[Mapping[str, Any]] = ()) -> Path:
    """Atomically persist a fully bound observed screen; never writes UNKNOWN defaults."""
    if not _KEEP_ID.fullmatch(keep_id) or not _SHA256.fullmatch(campaign_manifest_sha256):
        raise ValueError("keep/campaign identity is malformed")
    if not Path(source_path).is_absolute() or not _SHA256.fullmatch(source_sha256):
        raise ValueError("screen source identity is malformed")
    try:
        observed = datetime.fromisoformat(str(observed_at).replace("Z", "+00:00"))
        if observed.tzinfo is None:
            raise ValueError("screen observation time must have a timezone")
        if (observed - datetime.now(timezone.utc)).total_seconds() > FUTURE_SKEW_TOLERANCE_S:
            raise ValueError("screen observation timestamp is too far in the future")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"screen observation timestamp is malformed: {exc}") from exc
    try:
        source_file = Path(source_path)
        fd = os.open(source_file, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC)
        with os.fdopen(fd, "rb") as stream:
            before = os.fstat(stream.fileno())
            if not stat.S_ISREG(before.st_mode) or before.st_size > MAX_SOURCE_BYTES:
                raise ValueError("screen source is not a bounded regular file")
            hasher = hashlib.sha256()
            while chunk := stream.read(1024 * 1024):
                hasher.update(chunk)
            after = os.fstat(stream.fileno())
        if (before.st_dev, before.st_ino, before.st_mtime_ns, before.st_size) != (
                after.st_dev, after.st_ino, after.st_mtime_ns, after.st_size):
            raise ValueError("screen source changed while hashing")
    except OSError as exc:
        raise ValueError("screen source path is not readable") from exc
    if hasher.hexdigest() != source_sha256:
        raise ValueError("screen source bytes do not match the recorded SHA-256")
    body = {"schema": RECORD_SCHEMA, "keep_id": keep_id, "campaign_id": campaign_id,
            "campaign_manifest_sha256": campaign_manifest_sha256,
            "workload_signatures": dict(workload_signatures),
            "source_path": source_path, "source_sha256": source_sha256,
            "observed_at": observed_at, "screen": dict(screen),
            "rows": [dict(row) for row in rows],
            "refusals": [dict(row) for row in refusals]}
    body["record_sha256"] = _digest(body)
    encoded = _canonical(body) + b"\n"
    if len(encoded) > MAX_RECORD_BYTES:
        raise ValueError("screen record exceeds the bounded publication size")
    # Validate exact binding before publishing. A workload set must be supplied
    # by the caller from the resolved campaign; this writer does not infer it.
    workload_map = dict(body["workload_signatures"])
    if (not workload_map or any(not isinstance(key, str) or not key
            or not isinstance(value, str) or not _SHA256.fullmatch(value)
            for key, value in workload_map.items())):
        raise ValueError("workload IDs and signatures are required")
    if _validate_record(body, keep_id=keep_id, workloads=workload_map,
                        campaign_id=campaign_id,
                        manifest_digest=campaign_manifest_sha256) is None:
        raise ValueError("screen record failed its schema, hash, or row-binding checks")
    target = record_path(store_root, keep_id)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        current = _read_record(target)
        if current.get("record_sha256") == body["record_sha256"]:
            return target
        raise FileExistsError(f"immutable keep screen already exists: {target}")
    fd, temporary = tempfile.mkstemp(prefix=".akx-screen-", dir=target.parent)
    published = False
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(encoded.decode("ascii"))
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, target)
        published = True
        dirfd = os.open(target.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(dirfd)
        finally:
            os.close(dirfd)
    finally:
        if not published:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
        else:
            os.unlink(temporary)
    return target


def project(store_root: Path | str, *, keeps: Sequence[str],
            resolved_targets: Sequence[Mapping[str, Any]] | None,
            campaign_id: str | None, manifest_digest: str | None,
            manifest_path: str | None, now: datetime | None = None,
            stale_after_s: int = 1800) -> dict[str, Any]:
    """Project current real keep IDs against a resolved roster and immutable records."""
    now = now or datetime.now(timezone.utc)
    stamp = now.isoformat().replace("+00:00", "Z")
    if (not keeps or len(keeps) > MAX_KEEPS
            or any(not isinstance(item, str) or not _KEEP_ID.fullmatch(item) for item in keeps)):
        keeps = []
    workloads: dict[str, dict[str, str]] = {}
    if resolved_targets is not None and len(resolved_targets) <= MAX_WORKLOADS:
        for target in resolved_targets:
            if not isinstance(target, Mapping):
                continue
            ids = target.get("target_ids")
            signature = target.get("workload_signature")
            if (not isinstance(ids, (list, tuple)) or not isinstance(signature, str)
                    or not _SHA256.fullmatch(signature)):
                continue
            for workload_id in ids:
                if isinstance(workload_id, str) and workload_id:
                    workloads[workload_id] = {"workload_signature": signature,
                                              "status": str(target.get("status") or "unknown")}
    envelope = {"schema": STATUS_SCHEMA, "generated_at": stamp,
        "state": "fresh", "freshness": {"state": "fresh", "age_s": 0.0,
        "stale_after_s": stale_after_s},
        "source": {"kind": "resolved_campaign", "campaign_id": campaign_id,
                   "manifest_path": manifest_path,
                   "manifest_sha256": manifest_digest},
        "workloads": [{"workload_id": key, **value}
                      for key, value in sorted(workloads.items())], "keeps": []}
    if (not campaign_id or not manifest_digest or not _SHA256.fullmatch(manifest_digest)
            or not manifest_path or not Path(manifest_path).is_absolute()
            or not workloads):
        envelope["state"] = "missing_manifest"
        envelope["freshness"] = {"state": "absent", "age_s": None,
                                 "stale_after_s": stale_after_s}
        envelope["source"]["detail"] = "no resolved campaign workload manifest is available"
        envelope["keeps"] = [{"keep_id": keep_id, "source_path": None,
            "record_sha256": None,
            "freshness": {"state": "absent", "age_s": None,
                          "stale_after_s": stale_after_s,
                          "detail": "workload set is UNKNOWN until a resolved campaign manifest is published"},
            "rows": [], "verdict": "UNKNOWN", "refusal_remedy": None}
            for keep_id in keeps]
        return envelope

    root = Path(store_root)
    for keep_id in keeps:
        path = record_path(root, keep_id)
        loaded = None
        error = None
        malformed = False
        try:
            body = _read_record(path)
            loaded = _validate_record(body, keep_id=keep_id,
                workloads={key: value["workload_signature"]
                           for key, value in workloads.items()},
                campaign_id=campaign_id, manifest_digest=manifest_digest)
            if loaded is None:
                error = "record schema, digest, source, or workload binding did not validate"
                malformed = True
        except FileNotFoundError:
            error = "no AKX screen has been recorded for this retained keep"
        except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
            error = f"screen record is malformed: {type(exc).__name__}: {exc}"
            malformed = True
        age_s = None
        fresh = False
        if loaded:
            try:
                created = datetime.fromisoformat(str(loaded["observed_at"]).replace("Z", "+00:00"))
                if created.tzinfo is None:
                    raise ValueError("screen observation timestamp has no timezone")
                age_s = (now - created).total_seconds()
                if age_s < -FUTURE_SKEW_TOLERANCE_S:
                    raise ValueError("screen observation timestamp is too far in the future")
                age_s = max(0.0, age_s)
                fresh = age_s <= stale_after_s
            except (TypeError, ValueError, OverflowError):
                error = "screen observation timestamp is malformed or too far in the future"
                malformed = True
                loaded = None
        rows = []
        for workload_id, identity in sorted(workloads.items()):
            row = loaded["rows"].get(workload_id) if loaded else None
            classification = row["classification"] if row and fresh else "UNKNOWN"
            refusal = next((item for item in loaded["refusals"]
                            if item["workload_id"] == workload_id), None) if loaded else None
            rows.append({"workload_id": workload_id,
                "workload_signature": identity["workload_signature"],
                "classification": classification,
                "tier": row.get("tier") if row and fresh else None,
                "witnesses": row.get("witnesses") if row and fresh else None,
                "unknown_reason": (row.get("unknown_reason") if row and fresh else
                    None if loaded and fresh else error or "screen record is stale"),
                "verdict": refusal.get("verdict") if refusal and fresh else "UNKNOWN",
                "refusal_remedy": refusal.get("remedy") if refusal and fresh else None,
                "source_row_sha256": _digest(row) if row and fresh else None})
        envelope["keeps"].append({"keep_id": keep_id, "source_path": str(path),
            "record_sha256": loaded["record_sha256"] if loaded else None,
            "freshness": {"state": "fresh" if fresh else
                          "stale" if loaded else "malformed" if malformed else "absent",
                          "age_s": age_s, "stale_after_s": stale_after_s,
                          "detail": error or ("record is older than its staleness bound"
                              if not fresh else "current")},
            "rows": rows})
    return envelope
