"""AKU-10d2: the controller's maintenance caller, view -> bind -> execute.

One call collects the native catalog view outside the mutex, prepares and owner-binds
at most one retention job (``CandidateTransactions.prepare_retention``), then executes
it through the concrete ``CampaignMaintenanceBackend`` with the injected hold
provider. Without a real provider it refuses before collecting anything: absence of
the broker never becomes an eligibility bypass. No job (nothing expirable) is a
normal, side-effect-free outcome.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from . import maintenance_execution as me


class MaintenanceCallerRefused(RuntimeError):
    pass


@dataclass(frozen=True)
class MaintenanceRun:
    status: str  # "executed" | "nothing_to_expire"
    previews: tuple
    result: me.MaintenanceExecutionResult | None


def run_maintenance(transactions, policy, *, provider: Any, limit: int = 64,
                    now=None) -> MaintenanceRun:
    if provider is None or type(provider) in {me.MaintenanceHoldProvider, me.BrokerHoldProvider} \
            or not isinstance(provider, me.MaintenanceHoldProvider):
        raise MaintenanceCallerRefused("maintenance hold provider is unavailable")
    job, previews = transactions.prepare_retention(policy, limit=limit, now=now)
    if job is None:
        return MaintenanceRun("nothing_to_expire", tuple(previews), None)
    backend = me.CampaignMaintenanceBackend(
        transactions.controller, transactions.last_prepared, policy, provider)
    return MaintenanceRun("executed", tuple(previews), me.execute(job, backend=backend, now=now))
