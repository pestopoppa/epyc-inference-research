#!/usr/bin/env python3
"""The supported new-anchor epoch path (DS41-C126 gap a/b).

ORIGIN. Relaunching an AK lane on a new champion anchor (b0ba1d427) failed at startup:
``accumulate.BundleRecoveryRequired("invalid tip/anchor ancestry: persisted tip
dd6c9cdbdf85 is not an ancestor of anchor b0ba1d427835")``. That refusal is CORRECT --
`accumulate.load_bundle` must never silently move a store's champion-of-record/tip
lineage onto an anchor history it does not descend from, because the only other way a
tip could land outside the anchor's ancestry is a corrupt or tampered journal, and the
two cases are indistinguishable from inside `load_bundle` alone. But the refusal left no
SUPPORTED way forward for the genuine case -- a new epoch, the champion branch moved to
an anchor that supersedes the old lineage on purpose -- so the 2026-10-05 relaunch
worked around it with a throwaway fresh store dir per anchor (`store-b0ba1d427`),
abandoning the old store's journal history with no record of why.

THIS MODULE is that supported path. It is reachable only through an explicit
``--new-anchor-epoch`` CLI flag on the loop entry point (never a default, never inferred
from the refusal alone): when the flag is given AND the store would actually refuse the
requested anchor, it ARCHIVES the existing bundle projection (`accumulator-bundle.json`)
and the journal directory under the store by RENAME to a timestamped path -- nothing is
ever deleted -- and then initializes a fresh bundle with
``champion_of_record = tip = anchor_commit``, mirroring the existing fresh-store
initialization in `run.py` (the `restored = accumulate.Bundle(champion_of_record=
anchor_commit, tip=anchor_commit); restored.save(args.store)` branch guarding
experimental-serving's first durable bundle).

`would_refuse_anchor` (accumulate.py) is the companion PRE-FLIGHT: "would this store
refuse this anchor" without taking the flag, without building anything, and without
claiming a CPU region lock -- so a relaunch can check before paying for either.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import shutil

from . import accumulate

#: Archived bundle/journal pairs live under the store, never outside it and never
#: deleted, so the old epoch's full provenance stays inspectable next to the new one.
ARCHIVE_DIRNAME = "archived-epochs"


def _archive_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def start_new_anchor_epoch(store: Path, *, anchor_commit: str, is_ancestor,
                           stamp: str | None = None) -> tuple[accumulate.Bundle, str]:
    """Archive the store's current bundle projection + journal, then initialize a fresh
    bundle at ``champion_of_record = tip = anchor_commit``.

    Refuses (``ValueError``) unless `accumulate.would_refuse_anchor` says the store
    actually refuses this anchor first -- this is a repair for the ancestry-refusal case,
    never a casual reset switch, and a caller must not reach for it speculatively.
    Nothing is ever deleted: the prior ``accumulator-bundle.json`` and ``journal/`` are
    moved (renamed), never removed, under
    ``store/archived-epochs/<UTC timestamp>/``.
    """
    store = Path(store)
    would_refuse, reason = accumulate.would_refuse_anchor(
        store, anchor_commit=anchor_commit, is_ancestor=is_ancestor)
    if not would_refuse:
        raise ValueError(
            "--new-anchor-epoch refused: the store does not refuse this anchor "
            f"({reason}); this flag exists only for the ancestry-refusal case, never as "
            "a reset switch for a store that already accepts the anchor")

    archive_root = store / ARCHIVE_DIRNAME / (stamp or _archive_stamp())
    archive_root.mkdir(parents=True, exist_ok=False)

    archived = []
    bundle_path = store / accumulate.Bundle.FILENAME
    if bundle_path.exists():
        shutil.move(str(bundle_path), str(archive_root / accumulate.Bundle.FILENAME))
        archived.append(accumulate.Bundle.FILENAME)
    journal_path = store / accumulate.JOURNAL_DIRNAME
    if journal_path.exists():
        shutil.move(str(journal_path), str(archive_root / accumulate.JOURNAL_DIRNAME))
        archived.append(accumulate.JOURNAL_DIRNAME)

    fresh = accumulate.Bundle(champion_of_record=anchor_commit, tip=anchor_commit)
    fresh.save(store)
    note = (f"new anchor epoch at {anchor_commit[:12]}: archived "
            f"{', '.join(archived) if archived else 'nothing (store had no prior state)'} "
            f"to {archive_root} ({reason})")
    return fresh, note
