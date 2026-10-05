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
claiming a CPU region lock -- so a relaunch can check before paying for either. This
module itself classifies with `accumulate.peek_refusal` instead (lock-free, no
read-only shared-lock wait), because `would_refuse_anchor`'s populated-store branch
takes a BLOCKING shared lock -- correct for an external, patient pre-flight, but it
would deadlock a live-owner check here: a live owner holds the lock EXCLUSIVELY, so a
subsequent blocking shared-lock request from this same process (a different open file
description) would hang forever waiting for a release that an exclusive non-blocking
probe must instead report immediately.

The archive step itself takes the SAME exclusive journal lock `load_bundle`/`Bundle.save`
use for writes, NON-BLOCKING, for the whole check-under-lock + archive + init: a live
loop process holding that lock makes the flag refuse immediately rather than race a
concurrent writer out from under itself, or hang (review finding, 2026-10-05: the first
cut moved `journal/` with no lock and no live-owner check at all).

The trigger is narrowed to `accumulate.REFUSAL_KIND_ANCESTRY` specifically -- a corrupt
or unreadable journal, a symlinked store, or any other `BundleRecoveryRequired` reason
still refuses even with the flag given, because only an ancestry mismatch is evidence of
a genuine new epoch; every other reason is evidence the journal cannot be trusted, which
this module must never paper over.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import shutil

from .. import journal
from . import accumulate

#: Archived bundle/journal pairs live under the store, never outside it and never
#: deleted, so the old epoch's full provenance stays inspectable next to the new one.
ARCHIVE_DIRNAME = "archived-epochs"


def _archive_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def _not_ancestry_refusal(reason: str, kind: str) -> ValueError:
    return ValueError(
        "--new-anchor-epoch refused: the store's refusal is not an ancestry mismatch "
        f"(kind={kind!r}, reason={reason!r}); a new-anchor epoch only repairs the "
        "'persisted tip is not an ancestor of anchor' case. Inspect and restore the "
        "authoritative accumulator journal and its evidence before restarting -- "
        "--new-anchor-epoch is not a repair for a corrupt or unreadable journal.")


def _classify_or_raise(store: Path, *, anchor_commit: str, is_ancestor) -> str:
    """Lock-free classification shared by the pre-check and the under-lock re-check.
    Returns the ancestry-refusal reason, or raises the matching refusal ValueError."""
    exc = accumulate.peek_refusal(store, anchor_commit=anchor_commit, is_ancestor=is_ancestor)
    if exc is None:
        raise ValueError(
            "--new-anchor-epoch refused: the store does not refuse this anchor; this "
            "flag exists only for the ancestry-refusal case, never as a reset switch "
            "for a store that already accepts the anchor")
    if exc.kind != accumulate.REFUSAL_KIND_ANCESTRY:
        raise _not_ancestry_refusal(exc.reason, exc.kind)
    return exc.reason


def start_new_anchor_epoch(store: Path, *, anchor_commit: str, is_ancestor,
                           stamp: str | None = None,
                           dry_run: bool = False) -> tuple[accumulate.Bundle | None, str]:
    """Archive the store's current bundle projection + journal, then initialize a fresh
    bundle at ``champion_of_record = tip = anchor_commit``.

    Refuses (``ValueError``) unless the store's refusal for this anchor is specifically
    `accumulate.REFUSAL_KIND_ANCESTRY` -- never a casual reset switch, and never a repair
    for a corrupt/unreadable journal reported under any other kind. Takes the store's
    journal write lock (non-blocking) for the whole check-under-lock + archive + init and
    refuses if the store has a live owner (the lock is already held); nothing is ever
    deleted -- the prior ``accumulator-bundle.json`` and ``journal/`` are moved (renamed),
    never removed, under ``store/archived-epochs/<UTC timestamp>/``.

    ``dry_run=True`` only re-confirms the ancestry-refusal classification (lock-free,
    via `accumulate.peek_refusal`) and returns the action it WOULD take as a note, with
    ``bundle=None`` -- it never takes the lock and never touches disk.
    """
    store = Path(store)
    reason = _classify_or_raise(store, anchor_commit=anchor_commit, is_ancestor=is_ancestor)

    if dry_run:
        return None, (
            f"would start new anchor epoch at {anchor_commit[:12]}, archiving "
            f"{accumulate.Bundle.FILENAME} + {accumulate.JOURNAL_DIRNAME}/ under "
            f"{store / ARCHIVE_DIRNAME} ({reason})")

    book = journal.Journal(str(store / accumulate.JOURNAL_DIRNAME))
    lock_cm = book.write_lock(blocking=False)
    try:
        lock_cm.__enter__()
    except (BlockingIOError, OSError) as exc:
        raise ValueError(
            "--new-anchor-epoch refused: the store has a live owner (its journal write "
            f"lock is held): {exc}. Stop the owning process, or wait for it to release "
            "the store, before starting a new anchor epoch.") from exc
    try:
        # Re-classify UNDER the lock: the pre-check above ran lock-free and the
        # store's state may have moved since. `peek_refusal` (via `_classify_or_raise`)
        # takes no lock of its own -- a second flock from this same process on the SAME
        # lock file (a different open file description) could otherwise deadlock the
        # process against itself, since flock ownership is per open-file-description,
        # not per-process.
        reason = _classify_or_raise(store, anchor_commit=anchor_commit, is_ancestor=is_ancestor)

        archive_root = store / ARCHIVE_DIRNAME / (stamp or _archive_stamp())
        archive_root.mkdir(parents=True, exist_ok=False)

        archived = []
        bundle_path = store / accumulate.Bundle.FILENAME
        if bundle_path.exists():
            shutil.move(str(bundle_path),
                       str(archive_root / accumulate.Bundle.FILENAME))
            archived.append(accumulate.Bundle.FILENAME)
        journal_path = store / accumulate.JOURNAL_DIRNAME
        if journal_path.exists():
            # Renames the directory our own open lock fd lives under; flock
            # operates on the fd, not the path, so the lock we hold is unaffected
            # and is released cleanly when this function returns.
            shutil.move(str(journal_path),
                       str(archive_root / accumulate.JOURNAL_DIRNAME))
            archived.append(accumulate.JOURNAL_DIRNAME)

        fresh = accumulate.Bundle(champion_of_record=anchor_commit, tip=anchor_commit)
        fresh.save(store)
        note = (f"new anchor epoch at {anchor_commit[:12]}: archived "
                f"{', '.join(archived) if archived else 'nothing (store had no prior state)'} "
                f"to {archive_root} ({reason})")
        return fresh, note
    finally:
        lock_cm.__exit__(None, None, None)
