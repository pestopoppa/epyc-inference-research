"""Historical-only view when a newer bundle schema prevents rollback.

This inspection never returns a Bundle and cannot authorize admissions or repair
state. The last compatible snapshot is identified as historical; normal recovery
continues to refuse the newer record.
"""
from __future__ import annotations

import copy
import json
import os
import stat
from pathlib import Path
from typing import Any

from .. import journal, schemas
from . import accumulate


def inspect_compatible_bundle(store: Path, *, surface: str | None = None) -> dict[str, Any]:
    """Read the last compatible snapshot before an unsupported newer snapshot."""
    store = Path(store)
    if store.is_symlink():
        raise accumulate.BundleRecoveryRequired("bundle store cannot be a symlink")
    _, journal_dir = accumulate._surface_paths(store, surface)
    root = store / journal_dir
    if root.is_symlink() or not root.is_dir():
        raise accumulate.BundleRecoveryRequired("bundle journal is missing or symlinked")
    book = journal.Journal(str(root))
    with accumulate._existing_shared_lock(book):
        try:
            for shard in book.shards():
                info = os.lstat(shard.path)
                if (not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode)
                        or info.st_uid != os.getuid() or info.st_nlink != 1):
                    raise accumulate.BundleRecoveryRequired("journal shard identity is unsafe")
            report = book.scan()
        except (OSError, ValueError, TypeError, journal.JournalCorruption) as exc:
            raise accumulate.BundleRecoveryRequired(f"journal read failure: {exc}") from exc
        if report.torn_tail is not None:
            raise accumulate.BundleRecoveryRequired("journal has a torn tail")
        unknown = None
        if report.defects:
            if any(item.reason != "malformed envelope field(s)" or item.detail != "kind"
                   for item in report.defects):
                raise accumulate.BundleRecoveryRequired("journal has defects")
            first = min(report.defects, key=lambda item: (item.shard_index, item.line_number))
            shard = next((item for item in book.shards() if item.index == first.shard_index), None)
            if shard is None:
                raise accumulate.BundleRecoveryRequired("unknown event shard is missing")
            try:
                with open(shard.path, "rb") as stream:
                    raw = stream.readlines()[first.line_number - 1]
                envelope = json.loads(raw)
                seq, kind = envelope["seq"], envelope["kind"]
                if (not isinstance(seq, int) or isinstance(seq, bool) or seq < 1
                        or not isinstance(kind, str) or not kind):
                    raise ValueError("unknown event identity is malformed")
            except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
                raise accumulate.BundleRecoveryRequired("unknown event is malformed") from exc
            unknown = ((first.shard_index, first.line_number), seq)
        latest = None
        boundary = None
        previous_seq = 0
        for entry in sorted(report.entries, key=lambda item: (item.shard_index, item.line_number)):
            if unknown is not None and (entry.shard_index, entry.line_number) >= unknown[0]:
                continue
            if entry.seq <= previous_seq:
                raise accumulate.BundleRecoveryRequired("journal sequence is not strictly increasing")
            previous_seq = entry.seq
            if entry.kind != journal.KIND_LOOP_BUNDLE_SAVED:
                # Unknown or invalid non-bundle events cannot be silently skipped.
                schema = journal.SCHEMA_BOUND_KINDS.get(entry.kind)
                violations = (schemas.validate_record(entry.payload) if schema else
                              journal._validate_native_payload(entry.kind, entry.payload))
                if violations:
                    raise accumulate.BundleRecoveryRequired("invalid journal event")
                continue
            row = entry.payload
            snapshot = row.get("snapshot")
            if not isinstance(snapshot, dict):
                raise accumulate.BundleRecoveryRequired("invalid journal snapshot")
            version = snapshot.get("schema")
            supported = {journal.LOOP_BUNDLE_SNAPSHOT_SCHEMA_V1,
                         journal.LOOP_BUNDLE_SNAPSHOT_SCHEMA_V2,
                         journal.LOOP_BUNDLE_SNAPSHOT_SCHEMA_V3}
            if version not in supported:
                prefix = "epyc.autokernel.accumulator_bundle.v"
                suffix = version[len(prefix):] if isinstance(version, str) and version.startswith(prefix) else ""
                if (not suffix.isdecimal() or int(suffix) <= 3
                        or not isinstance(row.get("provenance"), str)
                        or not row["provenance"]
                        or set(row) != {"schema", "snapshot", "snapshot_sha256", "provenance"}
                        or row["schema"] != journal.LOOP_BUNDLE_SAVED_SCHEMA
                        or row["snapshot_sha256"] != journal.loop_bundle_snapshot_digest(snapshot)):
                    raise accumulate.BundleRecoveryRequired("invalid unsupported snapshot")
                boundary = entry.seq
                break
            validity = snapshot.get("measurement_validity")
            if (version in {journal.LOOP_BUNDLE_SNAPSHOT_SCHEMA_V2,
                            journal.LOOP_BUNDLE_SNAPSHOT_SCHEMA_V3}
                    and isinstance(validity, str) and validity
                    and validity not in journal.LOOP_BUNDLE_MEASUREMENT_VALIDITIES):
                # An older engine may know the schema but not a newer validity
                # value. Verify the rest of the carrier, then expose only history.
                probe = copy.deepcopy(row)
                probe["snapshot"]["measurement_validity"] = "current_snapshot"
                probe["snapshot_sha256"] = journal.loop_bundle_snapshot_digest(probe["snapshot"])
                if (row.get("snapshot_sha256") != journal.loop_bundle_snapshot_digest(snapshot)
                        or journal.validate_loop_bundle_saved_payload(probe)):
                    raise accumulate.BundleRecoveryRequired("invalid unsupported validity")
                boundary = entry.seq
                break
            if journal.validate_loop_bundle_saved_payload(row):
                raise accumulate.BundleRecoveryRequired("invalid compatible snapshot")
            try:
                bundle = accumulate.Bundle.from_dict(snapshot)
            except (TypeError, ValueError) as exc:
                raise accumulate.BundleRecoveryRequired(f"invalid compatible snapshot: {exc}") from exc
            if surface is not None and bundle.surface != surface:
                raise accumulate.BundleRecoveryRequired("compatible snapshot has wrong surface")
            latest = (entry, copy.deepcopy(snapshot), row["provenance"])
        if boundary is None and unknown is not None:
            if unknown[1] <= previous_seq:
                raise accumulate.BundleRecoveryRequired("unknown event sequence is invalid")
            boundary = unknown[1]
        if boundary is None or latest is None:
            raise accumulate.BundleRecoveryRequired("no compatible snapshot before newer schema")
        entry, snapshot, provenance = latest
        return {"schema": "epyc.autokernel.bundle_history_view.v1",
                "historical_only": True, "admission_authority": False,
                "snapshot": snapshot, "snapshot_sha256": journal.loop_bundle_snapshot_digest(snapshot),
                "original_provenance": provenance,
                "snapshot_seq": entry.seq, "unsupported_seq": boundary}


def main(argv: list[str] | None = None) -> int:
    import argparse
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", required=True, type=Path)
    parser.add_argument("--surface", choices=("cpu", "gpu"))
    args = parser.parse_args(argv)
    try:
        view = inspect_compatible_bundle(args.store, surface=args.surface)
    except (accumulate.BundleRecoveryRequired, ValueError) as exc:
        parser.exit(1, f"historical bundle unavailable: {exc}\n")
    print(json.dumps(view, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
