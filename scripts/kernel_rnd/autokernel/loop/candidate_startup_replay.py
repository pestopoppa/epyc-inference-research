"""AKU-09g4: rebuild the trusted candidate projection cache at controller startup.

Journal replay restores candidate entries but not the replayer-owned cache, so a
restarted controller would report ``replay_verified`` False until someone called
``inspect()``. This runs the trusted replayer once at startup. A durable intent
without completion is cached as pending with no committed state (the projection
stays unknown); any other replay refusal leaves the cache unbuilt and lagging.
"""
from __future__ import annotations


def rebuild_cache(controller) -> bool:
    """Return True when the startup replay rebuilt the cache from durable entries."""
    if not controller._candidate_entries or controller._candidate_projection_cache is not None:
        return False
    from . import candidate_transactions as ct
    try:
        ct.CandidateTransactions(controller).inspect()
    except ct.CandidateTransactionError:
        return False
    return controller._candidate_projection_cache is not None
