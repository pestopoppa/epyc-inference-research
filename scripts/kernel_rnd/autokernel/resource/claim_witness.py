#!/usr/bin/env python3
"""claim_witness.py — the seam between the GPU device claim and everything that reads it.

WHY THIS MODULE EXISTS
----------------------
`device_claim.py` (§14 AK2) ACQUIRES the MI210 claim. `preflight.py` (§3.5) and
the evaluation record (§7.4) both READ it. The three were written against the
same design and against each other's descriptions, and they did not meet:

  * **`preflight.claim_witness_preflight` needs a `GpuClaimReader`** —
    `Callable[[], Iterable[GpuClaimWitness]]` — and there was none. A GPU-scoped
    preflight therefore returned COULD_NOT_CHECK forever *even on a host where
    the claim was held*, and its reason text still said the substrate "does not
    exist yet (§2.5)" months after AK2 built it. Every call site would have
    hand-rolled the bridge, and the obvious hand-roll is wrong: `ClaimReceipt.
    holder_label` is `Optional[str]` while `GpuClaimWitness.holder_label` is
    `str`, so the natural one-liner produced a FAIL whose `whose` read literally
    `"None (pid 8800, via ...)"` — measured, before `GpuClaimWitness.__post_init__`
    started refusing it.

  * **`evaluation_event.resource_claim_receipt` is an opaque string.**
    `schemas.validate_evaluation_event` requires a non-empty string and nothing
    more; `device_claim.check_device_claim_held` needs `{claim_id, device_id}`.
    An event therefore cited an exclusivity receipt that NOTHING downstream could
    resolve: the id names a claim, the event does not name the device, and the
    two facts never met. `resolve_claim_receipt` closes it by resolving the id
    against the claim journal, which already records the full receipt dict on
    every `claim_acquired`.

WHAT THIS MODULE GUARANTEES
---------------------------
1. **Silence is never freedom.** A device whose claim state cannot be
   established raises `PreflightUnavailable`, which `claim_witness_preflight`
   triages into COULD_NOT_CHECK. The reader never returns `[]` for "I could not
   look" — that is the single fail-open shape that would fabricate a P-GPU-1
   precondition, and §2.6/§10.4 exist because `gpu_idle()` did exactly that.
2. **Every witness is attributable.** A held claim with an unreadable holder
   block is unverifiable, not free, and it is reported as such.
3. **Three outcomes, everywhere.** `check_event_claim_receipt` returns
   PASS / FAIL / COULD_NOT_CHECK. "The claim journal is unreadable" is not a
   failing receipt and is not a passing one.
4. **This module observes; it never acquires and never signals.** Acquisition
   lives in `device_claim.acquire_device_claim`, and invariant 9 stands: a PASS
   here is an observation, never a claim.

No inference, no benchmark, no process is started, stopped, or signalled.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, List, Mapping, Optional

from . import device_claim as _dc
from . import preflight as _pf

__all__ = [
    "WITNESS_SOURCE",
    "device_claim_witnesses",
    "device_claim_witness_reader",
    "gpu_claim_sources",
    "resolve_claim_receipt",
    "check_event_claim_receipt",
]

# What `GpuClaimWitness.source` says, so a journalled finding names the
# instrument that produced it rather than "gpu".
WITNESS_SOURCE = _dc.RECEIPT_SCHEMA


def _witness_label(payload: Mapping[str, Any]) -> str:
    """A non-empty attribution for a held claim.

    `holder.label` is optional in the claim payload, so it can never be the only
    source of the witness label — see this module's docstring for what a `None`
    label did to a FAIL finding. Purpose and campaign are both REQUIRED and
    non-empty at acquisition time (`acquire_device_claim` refuses otherwise), so
    the fallback always says something true.
    """
    holder = payload.get("holder")
    if isinstance(holder, Mapping):
        label = holder.get("label")
        if isinstance(label, str) and label.strip():
            return label.strip()
    parts = []
    for key in ("campaign_id", "purpose"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            parts.append(value.strip())
    claim_id = payload.get("claim_id")
    if isinstance(claim_id, str) and claim_id.strip():
        parts.append(claim_id.strip())
    return " / ".join(parts) if parts else "unlabelled device claim"


def device_claim_witnesses(
    device_ids: Iterable[str],
    *,
    lock_root: Optional[str] = None,
) -> List[Any]:
    """The currently-held claims on `device_ids`, as `GpuClaimWitness` objects.

    Raises `preflight.PreflightUnavailable` — never returns a short list — when
    any device's state cannot be established. States and their treatment:

      * `free`                      -> no witness (the device is genuinely unclaimed)
      * `held` / `revoking`         -> a witness (a draining claim is still held)
      * `stale`                     -> a witness; a dead holder's claim is still
                                       occupying the device until it is
                                       reclaimed, and reporting it as free is how
                                       two processes end up on one card
      * `unverifiable`              -> RAISE
      * `locked_without_payload`    -> RAISE (something holds the flock and we
                                       cannot say what; that is not "free")

    The device ids are validated by `device_claim`, so a typo raises rather than
    silently witnessing nothing about a device that does not exist.
    """
    ids = list(device_ids)
    if not ids:
        raise ValueError(
            "device_claim_witnesses() over an empty device list would attest to "
            "nothing; name the devices the run needs"
        )
    witnesses: List[Any] = []
    for device_id in ids:
        try:
            state = _dc.inspect_device_claim(device_id, lock_root)
        except ValueError:
            # A malformed device id is a caller bug, not an unknown host state:
            # it must not become COULD_NOT_CHECK, which would let a typo'd device
            # read as "unwitnessable" forever.
            raise
        except (OSError, _dc.DeviceClaimError) as exc:
            raise _pf.PreflightUnavailable(
                f"cannot read the device claim for {device_id!r}: {exc}"
            ) from exc

        status = state.get("state")
        if status == "free":
            continue
        if status in ("unverifiable", "locked_without_payload"):
            raise _pf.PreflightUnavailable(
                f"device {device_id!r} claim state is {status!r} "
                f"({state.get('error') or state.get('holder_liveness_reason')}); "
                "an unreadable claim is not an unclaimed device"
            )
        payload = state.get("claim")
        if not isinstance(payload, Mapping):
            raise _pf.PreflightUnavailable(
                f"device {device_id!r} reports state {status!r} with no claim "
                "payload to attribute it to"
            )
        holder = payload.get("holder")
        holder_pid = holder.get("pid") if isinstance(holder, Mapping) else None
        if holder_pid is not None and (
            isinstance(holder_pid, bool) or not isinstance(holder_pid, int)
        ):
            raise _pf.PreflightUnavailable(
                f"device {device_id!r} claim names a non-integer holder pid "
                f"{holder_pid!r}; the claim is occupied but unattributable"
            )
        witnesses.append(_pf.GpuClaimWitness(
            device_id=device_id,
            holder_pid=holder_pid,
            holder_label=_witness_label(payload),
            source=WITNESS_SOURCE,
            acquired_at=payload.get("acquired_at"),
        ))
    return witnesses


def device_claim_witness_reader(
    device_ids: Iterable[str],
    *,
    lock_root: Optional[str] = None,
):
    """A `preflight.GpuClaimReader` over the AK2 device claim substrate."""
    frozen = tuple(device_ids)

    def _read():
        return device_claim_witnesses(frozen, lock_root=lock_root)

    return _read


def gpu_claim_sources(
    device_ids: Iterable[str],
    *,
    region_lock_dir: Optional[str] = None,
    lock_root: Optional[str] = None,
    environ: Optional[Mapping[str, str]] = None,
    **kwargs: Any,
):
    """`ClaimSources` wired to both claim planes.

    The CPU region-lock directory and the GPU device lock root are the SAME
    on-disk root by design (`cpu_region.*.lock` beside `gpu_device.*.lock`), so
    when the caller names one and not the other they default to each other
    rather than silently diverging — two roots is exactly how two repositories
    stop excluding each other.
    """
    resolved_lock_root = lock_root
    if resolved_lock_root is None and region_lock_dir is not None:
        resolved_lock_root = str(region_lock_dir)
    resolved_region_dir = region_lock_dir
    if resolved_region_dir is None:
        resolved_region_dir = (
            str(resolved_lock_root) if resolved_lock_root is not None
            else _pf.default_region_lock_dir(environ)
        )
    return _pf.ClaimSources(
        region_lock_dir=Path(resolved_region_dir),
        gpu_claim_reader=device_claim_witness_reader(
            device_ids, lock_root=resolved_lock_root
        ),
        **kwargs,
    )


# =============================================================================
# Resolving an evaluation event's `resource_claim_receipt`
# =============================================================================

def resolve_claim_receipt(claim_id: str, claim_journal: Any):
    """Resolve an opaque `resource_claim_receipt` id to the receipt it names.

    Returns a `device_claim.ClaimReceipt`, or None when the journal records no
    acquisition under that id. RAISES on an unreadable journal — "I could not
    read the record" and "the record says this never happened" are different
    answers and the caller must be able to tell them apart.
    """
    if not isinstance(claim_id, str) or not claim_id.strip():
        raise ValueError("claim_id must be a non-empty string")
    read_all = getattr(claim_journal, "read_all", None)
    if not callable(read_all):
        raise TypeError(
            "claim_journal must expose read_all() -> list[dict] (a "
            "device_claim.ClaimJournal); there is no default sink on purpose"
        )
    for record in read_all():
        if not isinstance(record, Mapping):
            raise _dc.DeviceClaimUnreadable(
                f"claim journal record is a {type(record).__name__}, not an object"
            )
        if record.get("kind") != _dc.KIND_ACQUIRED:
            continue
        detail = record.get("detail")
        if not isinstance(detail, Mapping):
            continue
        if detail.get("claim_id") != claim_id:
            continue
        receipt = detail.get("receipt")
        if not isinstance(receipt, Mapping):
            raise _dc.DeviceClaimUnreadable(
                f"claim_acquired record for {claim_id!r} carries no receipt object"
            )
        # `from_dict` refuses a partial or extended receipt, so a record that
        # round-trips into a DIFFERENT receipt raises instead of resolving.
        return _dc.ClaimReceipt.from_dict(receipt)
    return None


def check_event_claim_receipt(event: Mapping[str, Any], claim_journal: Any):
    """PASS / FAIL / COULD_NOT_CHECK on "does this event's receipt name a real claim?"

    This is the check `schemas.validate_evaluation_event` structurally cannot
    perform: it requires `resource_claim_receipt` to be a non-empty string, and a
    non-empty string is exactly what an invented receipt also is. Binding the
    number to the exclusivity that produced it needs the claim record, which
    lives in a different module and a different file.

    PASS — the id resolves to a `claim_acquired` record whose receipt names the
           same campaign as the event.
    FAIL — the id resolves to nothing (no such claim was ever acquired), or to a
           claim acquired for a different campaign.
    COULD_NOT_CHECK — the event carries no usable receipt or campaign id, or the
           claim journal cannot be read. Neither of those is a verdict about the
           receipt.

    NOTE ON SCOPE: this establishes that the claim EXISTED and whose it was. It
    does not establish that the claim was held for the whole measurement window —
    the event is journalled after the run, by which time the claim is normally
    released. `device_claim.check_device_claim_held` answers the live question,
    and it must be called DURING the measurement.
    """
    if not isinstance(event, Mapping):
        return _pf.Check(
            _pf.COULD_NOT_CHECK,
            (f"event is a {type(event).__name__}, not a mapping",),
        )
    claim_id = event.get("resource_claim_receipt")
    if not isinstance(claim_id, str) or not claim_id.strip():
        return _pf.Check(
            _pf.COULD_NOT_CHECK,
            ("event carries no resource_claim_receipt string to resolve",),
        )
    campaign_id = event.get("campaign_id")
    if not isinstance(campaign_id, str) or not campaign_id.strip():
        return _pf.Check(
            _pf.COULD_NOT_CHECK,
            ("event carries no campaign_id, so a receipt cannot be attributed",),
        )
    try:
        receipt = resolve_claim_receipt(claim_id, claim_journal)
    except (_dc.DeviceClaimError, OSError, ValueError) as exc:
        return _pf.Check(
            _pf.COULD_NOT_CHECK,
            (f"claim journal could not be read: {type(exc).__name__}: {exc}",),
        )
    if receipt is None:
        return _pf.Check(_pf.FAIL, (
            f"resource_claim_receipt {claim_id!r} names no acquisition in the "
            "claim journal: the event asserts a measurement taken under "
            "exclusivity that was never recorded",
        ))
    if receipt.campaign_id != campaign_id:
        return _pf.Check(_pf.FAIL, (
            f"resource_claim_receipt {claim_id!r} was acquired for campaign "
            f"{receipt.campaign_id!r}, not {campaign_id!r}",
        ))
    return _pf.Check(_pf.PASS, (
        f"receipt {claim_id!r} resolves to a claim on device "
        f"{receipt.device_id!r} acquired at {receipt.acquired_at} by "
        f"campaign {receipt.campaign_id!r}",
    ))
