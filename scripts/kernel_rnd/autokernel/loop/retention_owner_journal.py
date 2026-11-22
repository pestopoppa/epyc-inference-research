"""AKU-10f3: persist native retention owner records in an additive journal kind.

Owner records are bound to (seed digest, config generation) — not to the supervisor
incarnation — so a record journaled by one incarnation stays current after restart
and is restored on replay instead of re-derived. Streams without this kind (legacy)
replay exactly as before: the controller derives each class at startup in memory and
appends nothing. A restored record is still compared digest-for-digest with its
re-derivation by ``build_native_view``; persistence never widens admission.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from . import native_retention_catalog as catalog

SCHEMA = "epyc.autokernel.native_retention_owner_journal.v1"
FIELDS = {"schema", "owner_class", "record", "record_digest"}


class OwnerJournalRefused(ValueError):
    pass


def make_payload(record: catalog.OwnerRecord) -> dict[str, Any]:
    return {"schema": SCHEMA, "owner_class": record.owner_class,
            "record": catalog._plain(record.body()), "record_digest": record.record_digest}


def validate(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, Mapping) or set(payload) != FIELDS or payload["schema"] != SCHEMA:
        raise OwnerJournalRefused("retention owner journal fields differ")
    restore(payload)
    return dict(payload)


def restore(payload: Mapping[str, Any]) -> catalog.OwnerRecord:
    body = payload["record"]
    if (not isinstance(body, Mapping) or body.get("schema") != catalog.OWNER_RECORD_SCHEMA
            or body.get("owner_class") != payload["owner_class"]):
        raise OwnerJournalRefused("retention owner journal record differs")
    try:
        artifacts = tuple(catalog.CatalogArtifact.from_dict(item) for item in body["artifacts"])
        return catalog.OwnerRecord(
            body["owner_class"], tuple(body["generation"]), body["inputs_digest"], artifacts,
            tuple(body["gaps"]), payload["record_digest"], catalog._OWNER_TOKEN)
    except (catalog.NativeRetentionCatalogRefused, KeyError, TypeError) as exc:
        raise OwnerJournalRefused(f"retention owner journal record refused: {exc}") from exc


def replayed_records(payloads, generation) -> dict[str, catalog.OwnerRecord]:
    """Latest journaled record per class that is current for ``generation``."""
    latest = {item["owner_class"]: item for item in payloads}
    records = {name: restore(item) for name, item in latest.items()}
    return {name: item for name, item in records.items() if item.generation == tuple(generation)}


def startup_records(controller, payloads) -> dict[str, catalog.OwnerRecord]:
    """Restore current journaled classes; derive the rest in memory (legacy: all)."""
    if controller._retention_catalog_seed is None:
        return {}
    records = replayed_records(payloads, controller._retention_generation_locked())
    for owner_class in catalog.OWNER_CLASSES:
        if owner_class not in records:
            records[owner_class] = controller._derive_retention_owner_locked(owner_class)
    return records
