"""autokernel.surface — the AK6 operator-surface plane.

WHY THIS PACKAGE EXISTS
-----------------------
Everything else in AutoKernel produces evidence. This plane produces the ONE
thing an operator actually looks at between campaigns: the `/kernel` panel on the
hub. It exists as its own plane because the failure mode it must prevent is not a
measurement failure at all — it is a REPORTING failure, and the project has paid
for exactly one of those already:

> Today's `/kernel` page is **absence-tolerant over a missing directory** — it
> renders clean when its producer is dead, which is the exact shape of AutoPilot
> dying at trial 1302 and staying dead ~23 HOURS with every dashboard green.

The hub reads `KERNEL_DASHBOARD_JSON`, which pointed at
`/mnt/raid0/llm/tmp/mi210-build/campaign/kernel_dashboard.json` — a GITIGNORED
SCRATCH PATH THAT DOES NOT EXIST — and credited it to a "kernel_store.py export"
that is not part of AutoKernel. The panel was reading a file no live producer
writes, and rendered clean the entire time. Absence tolerance is still required
(the page must not crash on a missing producer) but it must RENDER the absence.

WHAT LIVES HERE
---------------
`dashboard_contract` derives `schemas.SCHEMA_KERNEL_DASHBOARD_V2` from the modules
that OWN each fact and writes it to one durable path. It is the only module in
AutoKernel outside `journal`/`storage`/`resource`/`controller` that writes at all,
and that single write is bounded by `_assert_exportable_destination`, which
refuses every human-only target using `packager.HUMAN_ONLY_TARGET_PATTERNS` as the
single source of truth.

Nothing here measures, builds, benchmarks, or runs inference, and nothing here
holds freeze or cutover authority.

Owning design: `epyc-root/handoffs/active/autokernel-research-loop.md`, phase AK6.
"""
from __future__ import annotations

from . import dashboard_contract

__all__ = ["dashboard_contract"]
