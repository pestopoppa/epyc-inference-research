"""Concurrency-aware backend for pre-warm NUMA deployments.

Routes single sessions to the full-speed (1×96t) instance for maximum
per-request throughput. When concurrent requests arrive, migrates KV state
from the full instance to a quarter (48t) instance and routes new requests
to idle quarters.

Pre-warm architecture:
    - 1 full-speed instance (96t, node-pinned) — best single-session speed
    - 4 quarter instances (48t each, NUMA-quarter-pinned) — concurrent slots

The full instance is ALWAYS running (weights in RAM, mlocked). Quarter
instances are ALWAYS running too. The only dynamic operation is KV state
save/restore on transition, using llama.cpp's slot save/restore API.

KV migration flow (Phase D):
    1. Session A starts → routes to full (96t) instance
    2. Session B arrives while Session A between turns (full idle)
       → Save A's KV from full (POST /slots/0?action=save)
       → Restore A's KV on quarter 0 (POST /slots/0?action=restore)
       → Route A's next turn to quarter 0
       → Route B to full instance (fresh, max speed)
    3. Session A completes → quarter 0 freed
    4. Only one session left → next turn goes back to full (max speed)

Usage:
    full_backend = CachingBackend(srv_96t, ...)
    quarter_backends = [CachingBackend(srv_48t_0, ...), ...]
    ca = ConcurrencyAwareBackend(full_backend, quarter_backends, role="frontdoor")
"""

from __future__ import annotations

import logging
import re
import threading
import time
from contextlib import contextmanager
from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from src.backends import kv_slot_files as _kv_slot_files
from src.backends import serving_calls
from src.runtime.quiescence import suppress as _quiescence_suppress
from src.scheduling import gate_observation

if TYPE_CHECKING:
    from src.scheduling.migration_transaction import MigrationTransaction

try:
    import httpx
    _HTTPX_AVAILABLE = True
except ImportError:
    _HTTPX_AVAILABLE = False

logger = logging.getLogger(__name__)

# Slot save/restore timeout — KV state can be 2-16 GB for production conversations
_SLOT_SAVE_TIMEOUT = 30.0  # seconds
_SLOT_RESTORE_TIMEOUT = 30.0

_STATE_UNASSIGNED = "unassigned"
_STATE_ASSIGNED_FULL = "assigned_full"
_STATE_MIGRATION_PENDING = "migration_pending"
_STATE_ASSIGNED_QUARTER = "assigned_quarter"
_STATE_MIGRATION_FAILED_COLD = "migration_failed_cold"


def _record_thrash_skip(reason: str) -> None:
    """Best-effort ``kv_migration_thrash_skipped_total`` increment (WP-4).

    A metrics failure must never break a dispatch/migration decision, so the
    counter import + increment are fully swallowed.
    """
    try:
        from src.metrics import migration_counters

        migration_counters.record_thrash_skip(reason)
    except Exception:  # pragma: no cover - metrics must never break dispatch
        pass


def _get_base_url(backend: Any) -> str | None:
    """Extract the base URL from a CachingBackend or LlamaServerBackend."""
    # CachingBackend wraps LlamaServerBackend; historical variants used
    # either `.backend` or `._backend`.
    inner = getattr(backend, "backend", None) or getattr(backend, "_backend", backend)
    config = getattr(inner, "config", None)
    if config:
        return getattr(config, "base_url", None)
    return None


def _extract_port(url: str | None) -> int | None:
    if not url:
        return None
    try:
        from urllib.parse import urlparse

        return urlparse(url).port
    except Exception:
        return None


def _shape_for_regions(regions: frozenset[str] | set[str] | list[str]) -> str:
    rs = set(regions or [])
    if rs == {"q0", "q1", "q2", "q3"}:
        return "full"
    if rs == {"q0", "q1"}:
        return "half0"
    if rs == {"q2", "q3"}:
        return "half1"
    if len(rs) == 1:
        return next(iter(rs))
    return "+".join(sorted(rs)) if rs else "unknown"


def _slot_filename(role: str, session_id: str, txn_id: str) -> str:
    """Return a llama-server slot-save filename accepted under --slot-save-path."""
    raw = f"{role}_{session_id}_{txn_id}"
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw).strip("._")
    return f"kv_migrate_{safe[:180] or 'session'}.bin"


def _reported_tokens(result: "bool | int | None") -> int | None:
    """Token count a slot helper reported, or None when it did not report one.

    None means "no usable count" — a transport failure, OR a caller (test double,
    older shim) that still answers with a bare ``True``. Both must skip the
    save/restore equality gate rather than fail it: an unknown count is not
    evidence of a mismatch. ``False`` is a failure. ``0`` is a real count and is
    NOT None — that is the whole point of this function.
    """
    if result is None or result is False:
        return None
    if result is True:          # legacy bool contract: success, count unknown
        return None
    return int(result)


def _slot_save(base_url: str, slot_id: int = 0, filename: str | None = None) -> int | None:
    """Save KV state from a llama-server slot.

    Returns the number of tokens the server reports saving, or None on failure.
    Returning the COUNT rather than a bool is load-bearing: a save that writes
    zero tokens is an HTTP 200, and callers must be able to tell it apart from a
    real one before they act on it destructively (see _migrate_kv).
    """
    if not _HTTPX_AVAILABLE:
        return None
    try:
        url = f"{base_url}/slots/{slot_id}?action=save"
        kwargs = {"json": {"filename": filename}} if filename else {}
        resp = httpx.post(url, timeout=_SLOT_SAVE_TIMEOUT, **kwargs)
        if resp.status_code == 200:
            data = resp.json()
            n_saved = int(data.get("n_saved", 0) or 0)
            logger.info(
                "KV save: slot %d, %d tokens, %.1fms",
                slot_id, n_saved, data.get("timings", {}).get("save_ms", 0),
            )
            return n_saved
        logger.warning("KV save failed: HTTP %d from %s", resp.status_code, url)
        return None
    except Exception as exc:
        logger.debug("KV save failed: %s", exc)
        return None


def _slot_restore(base_url: str, slot_id: int = 0, filename: str | None = None) -> int | None:
    """Restore KV state to a llama-server slot.

    Returns the number of tokens the server reports restoring, or None on
    failure. The count is what makes a restore verifiable: HTTP 200 proves the
    request was served, not that any KV came back, and _migrate_kv erases the
    SOURCE slot on the strength of this answer.
    """
    if not _HTTPX_AVAILABLE:
        return None
    try:
        url = f"{base_url}/slots/{slot_id}?action=restore"
        kwargs = {"json": {"filename": filename}} if filename else {}
        resp = httpx.post(url, timeout=_SLOT_RESTORE_TIMEOUT, **kwargs)
        if resp.status_code == 200:
            data = resp.json()
            n_restored = int(data.get("n_restored", 0) or 0)
            logger.info(
                "KV restore: slot %d, %d tokens, %.1fms",
                slot_id, n_restored, data.get("timings", {}).get("restore_ms", 0),
            )
            return n_restored
        logger.warning("KV restore failed: HTTP %d from %s", resp.status_code, url)
        return None
    except Exception as exc:
        logger.debug("KV restore failed: %s", exc)
        return None


def _slot_erase(base_url: str, slot_id: int = 0) -> bool:
    """Erase KV state from a llama-server slot."""
    if not _HTTPX_AVAILABLE:
        return False
    try:
        resp = httpx.post(f"{base_url}/slots/{slot_id}?action=erase", timeout=5.0)
        return resp.status_code == 200
    except Exception:
        return False


# Aborts after which no restore can still be reading the slot file: the save
# never produced a restorable file, or the server ANSWERED the restore (with a
# wrong count). `restore_failed` is deliberately absent — it includes a client
# timeout while the server may still be reading, so that file is left for the
# age-gated sweep.
_SLOT_FILE_DONE_ABORTS = frozenset({"save_failed", "restore_token_mismatch"})


def _slot_file_safe_to_unlink(transaction: Any) -> bool:
    """True once a migration's slot file can no longer be read by any restore.

    The file name embeds the per-transaction ``txn_id`` and a retry is a new
    transaction, so no retry path ever reuses a previous file.
    """
    if transaction is None:
        return False
    from src.scheduling.migration_transaction import MigrationState

    state = getattr(transaction, "state", None)
    if state in (
        MigrationState.VERIFIED,        # restore returned the saved count
        MigrationState.SOURCE_ERASED,
        MigrationState.COMMITTED,
    ):
        return True
    return (
        state is MigrationState.ABORTED
        and getattr(transaction, "detail", "") in _SLOT_FILE_DONE_ABORTS
    )


class ConcurrencyAwareBackend:
    """Routes requests between full-speed and quarter instances based on load.

    Single active request  → full-speed instance (max per-request throughput)
    Multiple active requests → quarter instances (max concurrency)

    KV state migration (Phase D): When the full instance is idle between turns
    and a new session arrives, the existing session's KV state is migrated from
    the full instance to a quarter instance. This is best-effort — if migration
    fails, the quarter starts cold (no KV state), which is functionally correct
    but loses prefix cache benefit.
    """

    def __init__(
        self,
        full_backend: Any,
        quarter_backends: list[Any],
        role: str = "",
        full_port: int = 0,
        topology_role: str | None = None,
        quarter_topology_idxs: list[int] | None = None,
        health_tracker: Any = None,
        native_batch_width: int = 1,
        slot_save_dir: "str | PathLike[str] | None" = None,
        kv_slot_policy: "_kv_slot_files.KvSlotRetentionPolicy | None" = None,
    ):
        if not quarter_backends:
            raise ValueError("ConcurrencyAwareBackend requires at least one quarter backend")
        self._full = full_backend
        self._quarters = quarter_backends
        self._role = role
        self._topology_role = topology_role or role
        self._full_port = full_port
        self._native_batch_width = max(1, int(native_batch_width))
        self._native_full_active = 0
        self._lock = threading.Lock()

        # WP-12 fleet layer: when this backend IS the fleet (one CAB per
        # physical fleet), the shared BackendHealthTracker is injected so
        # health is recorded per DISPATCHED endpoint — one circuit per
        # (fleet, port), one fact for every bound role. None (the legacy
        # per-role construction) leaves all health bookkeeping to the
        # primitives layer exactly as before.
        self._health_tracker = health_tracker

        # DISPATCH-A2: the TRUE NUMA_CONFIG topology index of each quarter
        # backend, resolved by PORT at construction. The dispatcher keys region
        # locks on topology index, so this MUST be the index whose cpuset matches
        # the quarter server's physical cores — NOT the quarter's list position.
        # They diverge whenever a misaligned `full:` endpoint is demoted into the
        # quarters pool (the full-stripped list no longer starts at topology idx
        # 1). Default (None) → legacy positional mapping (list idx i → topology
        # idx i+1), preserving the aligned full+quarters layout exactly.
        if quarter_topology_idxs is None:
            self._quarter_topology_idx = [i + 1 for i in range(len(quarter_backends))]
        else:
            if len(quarter_topology_idxs) != len(quarter_backends):
                raise ValueError(
                    "quarter_topology_idxs length "
                    f"({len(quarter_topology_idxs)}) != quarter_backends "
                    f"({len(quarter_backends)})"
                )
            self._quarter_topology_idx = list(quarter_topology_idxs)

        # Extract base URLs for slot API calls
        self._full_url = _get_base_url(full_backend)
        self._quarter_urls = [_get_base_url(q) for q in quarter_backends]

        # Tracking state
        self._full_active = False
        self._quarter_active: list[bool] = [False] * len(quarter_backends)
        self._total_requests = 0
        self._full_requests = 0
        self._quarter_requests = 0
        self._migrations = 0
        self._migration_failures = 0

        # Session affinity: track which session was last on the full instance
        # so we can migrate its KV state to a quarter on concurrent arrival.
        self._full_last_session: str | None = None

        # Migration tracking: session_id → quarter_idx
        # Only populated after restore succeeds or when the session is explicitly
        # pinned to a cold-start quarter following migration failure.
        self._session_quarter: dict[str, int] = {}
        self._session_state: dict[str, dict[str, Any]] = {}

        # WP-4 reverse-migration state: per-session last-seen timestamps + per-session
        # migration counts + in-flight guard to prevent double-firing.
        self._full_idle_since: float | None = None  # monotonic time when full last released
        self._session_last_seen: dict[str, float] = {}
        self._reverse_migration_counts: dict[str, int] = {}
        self._reverse_migration_in_flight: dict[str, bool] = {}

        # Phase D (cross-role-bw-aware-routing): topology-aware quarter ordering.
        # When the full instance is busy, try quarters in an order that prefers
        # NUMA-disjoint placement first. For frontdoor with full on NUMA_NODE0
        # (cores 0-47), the matrix shows q3 (Q1B 72-95) + q2 (Q1A 48-71)
        # beat q1 (Q0B 24-47) + q0 (Q0A 0-23) by 1.5×-1.7× on the
        # full+own-quarter test. See `_compute_quarter_preference()`.
        self._quarter_preference_order: list[int] = self._compute_quarter_preference()

        # DISPATCH-A alignment guard: confirm the endpoint wired into this
        # backend's "full" slot really is the topology's idx-0 (all-region)
        # instance. When the live wiring labels a quarter-sized endpoint as
        # `full:` (a 24-core quarter impersonating the 96-core full), routing
        # there would acquire idx-0's whole-machine region lock and serialize
        # every role. If the port we hold as full does not match NUMA_CONFIG's
        # idx-0 port for this role, we treat the full slot as misaligned and
        # never emit the full candidate (quarters only). Unknown role/port →
        # aligned (preserve legacy behavior). Computed once; static per backend.
        self._full_slot_aligned: bool = self._compute_full_slot_alignment()
        if self._full is not None and not self._full_slot_aligned:
            # A full backend is present but its port is not the topology idx-0
            # port (a quarter impersonating the full) AND it was not demoted
            # upstream (`_init_caching_backends` demotes it into the quarters pool
            # when the true index is resolvable; this branch remains as a safety
            # net for direct construction or an unresolvable port). The full
            # candidate is suppressed so the all-region lock is never grabbed;
            # that endpoint's own quarter capacity is NOT reclaimed here — prefer
            # demotion at the construction site to reclaim it.
            logger.warning(
                "ConcurrencyAwareBackend[%s]: full slot MISALIGNED — port %s is "
                "not the NUMA_CONFIG idx-0 port for topology role %s; the full "
                "(all-region) candidate is disabled to avoid a whole-machine "
                "lock grab (endpoint capacity is stranded — demote it at "
                "construction to reclaim it). Requests route to quarters only.",
                self._role or "unknown", self._full_port, self._topology_role,
            )

        # Phase E (cross-role-bw-aware-routing): KV migration status.
        # 2026-05-24 update: migration is NOW also wired into the per-region-
        # locks `_dispatch` path (the "port" the handoff asked for). The flag
        # now means "any path actually performs save/restore migration" and
        # is enabled whenever httpx is available. Operators can disable per-
        # instance via `ConcurrencyAwareBackend(..., kv_migration_disabled=True)`
        # for opt-out testing. Status reporting via `kv_migration_status()`
        # tells the dashboard which dispatch path is live.
        self._kv_migration_enabled = _HTTPX_AVAILABLE

        logger.info(
            "ConcurrencyAwareBackend[%s%s]: 1 full (%s) + %d quarters, "
            "quarter preference=%s, KV migration %s (per_region_locks=%s)",
            role or "unknown",
            (
                f" topology={self._topology_role}"
                if self._topology_role != self._role
                else ""
            ),
            self._full_url or "?",
            len(quarter_backends),
            self._quarter_preference_order,
            "enabled" if self._kv_migration_enabled else "disabled",
            self._per_region_locks_enabled(),
        )

        # Disk-leak audit 2026-09-27 item 3: slot files the servers write under
        # --slot-save-path. None → derived like the launcher does (cache_dir /
        # kv_slots / <role>); the policy is orchestration/kv_slot_retention_policy.yaml.
        self._slot_save_dir_override = Path(slot_save_dir) if slot_save_dir else None
        self._kv_slot_policy = (
            kv_slot_policy if kv_slot_policy is not None else _kv_slot_files.load_policy()
        )
        self._inflight_slot_files: set[str] = set()
        self._last_slot_sweep: float | None = None
        self._maybe_sweep_slot_files()  # startup sweep (report-only unless enabled)

    # === KV-migration slot files (disk-leak audit 2026-09-27) ===

    def _slot_save_dirs(self) -> list[Path]:
        if self._slot_save_dir_override is not None:
            return [self._slot_save_dir_override]
        return _kv_slot_files.default_slot_save_dirs(self._topology_role, self._role)

    def _slot_file_begin(self, filename: str) -> None:
        with self._lock:
            self._inflight_slot_files.add(filename)

    def _slot_file_end(self, filename: str, transaction: Any) -> None:
        """Terminal bookkeeping for one migration's slot file. Never raises."""
        try:
            with self._lock:
                self._inflight_slot_files.discard(filename)
            if not _slot_file_safe_to_unlink(transaction):
                if transaction is not None:
                    logger.info(
                        "kv slot file %s kept (state=%s detail=%s); left for the age-gated sweep",
                        filename,
                        getattr(getattr(transaction, "state", None), "value", None),
                        getattr(transaction, "detail", ""),
                    )
            elif self._kv_slot_policy.unlink_on_terminal:
                reason = (
                    f"migration_terminal txn={getattr(transaction, 'txn_id', '?')} "
                    f"state={transaction.state.value}"
                )
                for slot_dir in self._slot_save_dirs():
                    if _kv_slot_files.unlink_slot_file(slot_dir, filename, reason=reason):
                        break
            self._maybe_sweep_slot_files()
        except Exception as exc:  # pragma: no cover - hygiene must never break migration
            logger.warning("kv slot file bookkeeping failed for %s: %s", filename, exc)

    def _maybe_sweep_slot_files(self, force: bool = False) -> list[Any] | None:
        """Age-gated orphan sweep of this backend's slot dir(s). Never raises.

        Skipped entirely while ANY migration of this backend is in flight, and
        rate-limited to one pass per policy interval. With the sweep disabled by
        policy it runs report-only and warns about stale files instead.
        """
        try:
            policy = self._kv_slot_policy
            now = time.monotonic()
            with self._lock:
                if self._inflight_slot_files:
                    return None
                if (
                    not force
                    and self._last_slot_sweep is not None
                    and now - self._last_slot_sweep < policy.sweep_interval_s
                ):
                    return None
                self._last_slot_sweep = now
                in_flight = set(self._inflight_slot_files)
            results = []
            for slot_dir in self._slot_save_dirs():
                res = _kv_slot_files.sweep_orphan_slot_files(
                    slot_dir,
                    min_age_s=policy.sweep_min_age_s,
                    in_flight=in_flight,
                    max_files=policy.sweep_max_files,
                    dry_run=not policy.sweep_enabled,
                )
                results.append(res)
                if res.dry_run and res.stale:
                    logger.warning(
                        "kv slot dir %s holds %d stale kv_migrate_*.bin file(s) "
                        "(%.2f GiB) older than %.0f min; orphan sweep disabled by "
                        "orchestration/kv_slot_retention_policy.yaml",
                        res.slot_dir, len(res.stale), res.stale_bytes / 2**30,
                        policy.sweep_min_age_minutes,
                    )
                elif res.removed:
                    logger.info(
                        "kv slot sweep %s: removed %d orphan file(s), %.2f GiB%s",
                        res.slot_dir, len(res.removed), res.removed_bytes / 2**30,
                        " (capped; more remain)" if res.truncated else "",
                    )
            return results
        except Exception as exc:  # pragma: no cover - hygiene must never break serving
            logger.warning("kv slot sweep failed: %s", exc)
            return None

    def kv_migration_status(self) -> dict[str, Any]:
        """Operator-visible status of the KV-migration subsystem.

        Returns:
            {
              "enabled": bool,           # migration code is active in current dispatch path
              "per_region_locks": bool,  # which dispatch path is in effect
              "dispatch_path": str,      # "legacy_select" | "per_region_locks"
              "reason": str,             # human description (when disabled)
            }

        2026-05-24: migration is now ported into both dispatch paths. The
        per-region-locks `_dispatch` path kicks off the same async save/restore
        thread the legacy `_select` path uses, on the same conditions (full
        acquired by new session, old session has no quarter affinity yet).
        """
        path = "per_region_locks" if self._per_region_locks_enabled() else "legacy_select"
        return {
            "enabled": self._kv_migration_enabled,
            "per_region_locks": self._per_region_locks_enabled(),
            "dispatch_path": path,
            "reason": (
                "" if self._kv_migration_enabled else "httpx unavailable"
            ),
        }

    def _compute_quarter_preference(self) -> list[int]:
        """Return quarter indices ordered by preference when full is busy.

        Logic: quarters whose CPU cores are DISJOINT from the full instance's
        cores come first (no cpu-set contention with the in-flight full
        request), then quarters with partial overlap. Within each bucket,
        original numerical order is preserved.

        Falls back to `[0, 1, 2, ..., N-1]` if NUMA_CONFIG isn't importable
        (e.g. dev/test contexts).
        """
        try:
            from scripts.server.stack_numa import NUMA_CONFIG  # type: ignore[import-not-found]
            from src.runtime.instance_topology import parse_cpu_list
        except Exception:
            return list(range(len(self._quarters)))

        cfg = NUMA_CONFIG.get(self._topology_role) or NUMA_CONFIG.get(self._role)
        if not cfg or not cfg.get("instances"):
            return list(range(len(self._quarters)))

        instances = cfg["instances"]
        # Instance 0 is full; quarters are 1..N
        if len(instances) < 2:
            return list(range(len(self._quarters)))

        full_cores = parse_cpu_list(instances[0][0])

        disjoint: list[int] = []
        overlapping: list[int] = []
        for q_idx in range(len(self._quarters)):
            topo_idx = self._quarter_topology_idx[q_idx]  # TRUE NUMA_CONFIG index (port-resolved)
            if topo_idx >= len(instances):
                # More backends than topology entries — append at the end
                overlapping.append(q_idx)
                continue
            q_cores = parse_cpu_list(instances[topo_idx][0])
            if full_cores & q_cores:
                overlapping.append(q_idx)
            else:
                disjoint.append(q_idx)

        return disjoint + overlapping

    def _select(self, session_id: str = "") -> tuple[Any, int, bool]:
        """Select the best backend for the next request.

        Args:
            session_id: Optional session identifier for affinity routing.

        Returns (backend, index, is_full) where index is:
            -1 for full instance
            0..N for quarter instances
        """
        with self._lock:
            self._total_requests += 1

            # Check session affinity: if this session was migrated to a quarter,
            # route it back to that quarter (preserves KV state from migration).
            if session_id:
                if session_id in self._session_quarter:
                    q_idx = self._session_quarter[session_id]
                    if 0 <= q_idx < len(self._quarters):
                        self._quarter_active[q_idx] = True
                        self._quarter_requests += 1
                        self._set_session_state(
                            session_id,
                            state=_STATE_ASSIGNED_QUARTER,
                            quarter=q_idx,
                        )
                        return self._quarters[q_idx], q_idx, False
                pending = self._session_state.get(session_id)
                if pending and pending.get("state") in {
                    _STATE_MIGRATION_PENDING,
                    _STATE_MIGRATION_FAILED_COLD,
                }:
                    q_idx = int(pending.get("quarter", -1))
                    if 0 <= q_idx < len(self._quarters):
                        self._quarter_active[q_idx] = True
                        self._quarter_requests += 1
                        return self._quarters[q_idx], q_idx, False

            # If full instance is idle, use it (best per-request speed).
            # Quarters-only backends (self._full is None, e.g. a demoted
            # misaligned full) skip straight to quarter selection.
            if self._full is not None and not self._full_active:
                # If there was a previous session on full and we're a NEW session,
                # we need to migrate the previous session's KV to a quarter first.
                if (
                    self._full_last_session
                    and session_id
                    and session_id != self._full_last_session
                    and self._full_last_session not in self._session_quarter
                ):
                    # Find an idle quarter for the migration target
                    migrate_target = None
                    for i, active in enumerate(self._quarter_active):
                        if not active:
                            migrate_target = i
                            break

                    # INF-78 OAB-3 (R2): a quiescent_after request starts no KV-migration
                    # thread (it outlives the reply); it just takes full without migrating.
                    if migrate_target is not None and not _quiescence_suppress("kv_migration"):
                        old_session = self._full_last_session
                        self._migrations += 1
                        self._set_session_state(
                            old_session,
                            state=_STATE_MIGRATION_PENDING,
                            quarter=migrate_target,
                        )
                        self._full_active = True
                        self._full_requests += 1
                        self._full_last_session = session_id
                        self._set_session_state(
                            session_id,
                            state=_STATE_ASSIGNED_FULL,
                            quarter=None,
                        )

                        threading.Thread(
                            target=self._migrate_kv,
                            args=(old_session, migrate_target),
                            daemon=True,
                            name=f"kv-migrate-{self._role}-{old_session[:8]}",
                        ).start()

                        return self._full, -1, True

                self._full_active = True
                self._full_requests += 1
                if session_id:
                    self._full_last_session = session_id
                    self._set_session_state(
                        session_id,
                        state=_STATE_ASSIGNED_FULL,
                        quarter=None,
                    )
                return self._full, -1, True

            # Full is busy — find an idle quarter
            for i, active in enumerate(self._quarter_active):
                if not active:
                    self._quarter_active[i] = True
                    self._quarter_requests += 1
                    if session_id:
                        self._set_session_state(
                            session_id,
                            state=_STATE_ASSIGNED_QUARTER,
                            quarter=i,
                        )
                        self._session_quarter[session_id] = i
                    return self._quarters[i], i, False

            # All quarters busy — overflow to least-recently-used quarter
            idx = self._quarter_requests % len(self._quarters)
            self._quarter_active[idx] = True
            self._quarter_requests += 1
            if session_id:
                self._set_session_state(
                    session_id,
                    state=_STATE_ASSIGNED_QUARTER,
                    quarter=idx,
                )
                self._session_quarter[session_id] = idx
            logger.warning(
                "All %s instances busy (%d quarters), overflow to quarter %d",
                self._role, len(self._quarters), idx,
            )
            return self._quarters[idx], idx, False

    def _set_session_state(
        self,
        session_id: str,
        *,
        state: str,
        quarter: int | None,
        detail: str = "",
    ) -> None:
        if not session_id:
            return
        self._session_state[session_id] = {
            "state": state,
            "quarter": quarter,
            "detail": detail,
            "updated_at": time.time(),
        }

    def _finalize_quarter_assignment(
        self,
        session_id: str,
        quarter: int,
        *,
        state: str,
        detail: str = "",
    ) -> None:
        with self._lock:
            self._session_quarter[session_id] = quarter
            self._set_session_state(session_id, state=state, quarter=quarter, detail=detail)

    def _quarter_for_session_locked(self, session_id: str) -> int | None:
        """Return this session's quarter affinity or pending quarter.

        Caller must hold ``self._lock``. The per-region dispatch path needs to
        treat ``migration_pending`` and ``migration_failed_cold`` as reserved
        quarter assignments; otherwise concurrent handovers can start duplicate
        migrations before the first save/restore commits.
        """
        if not session_id:
            return None
        q_idx = self._session_quarter.get(session_id)
        if q_idx is not None:
            return q_idx
        state = self._session_state.get(session_id) or {}
        if state.get("state") in {
            _STATE_MIGRATION_PENDING,
            _STATE_ASSIGNED_QUARTER,
            _STATE_MIGRATION_FAILED_COLD,
        }:
            try:
                q_idx = int(state.get("quarter", -1))
            except (TypeError, ValueError):
                return None
            if 0 <= q_idx < len(self._quarters):
                return q_idx
        return None

    def _migrate_kv(
        self,
        session_id: str,
        target_quarter: int,
        transaction: Optional["MigrationTransaction"] = None,
    ) -> Optional["MigrationTransaction"]:
        """Migrate KV state from full instance to a quarter (background thread).

        WP-3 refactor: drives an explicit MigrationTransaction state machine
        (planned → saving → restoring → verified → source_erased → committed
        or → aborted) so callers can await completion + observe failure
        modes. Best-effort still — if save/restore fails, the transaction
        moves to ABORTED and the quarter starts cold.

        Args:
          session_id: KV cache owner.
          target_quarter: destination quarter index (0..N-1 in self._quarters).
          transaction: optional pre-allocated transaction (e.g. created by
            the dispatcher's load-transition trigger so it can wait on
            transaction.event). If None, an internal one is created and
            returned for telemetry-only consumers.

        Returns the (possibly-created) MigrationTransaction.
        """
        from src.scheduling.migration_transaction import (
            MigrationState,
            MigrationTransaction,
        )

        if not self._full_url:
            return transaction

        target_url = (
            self._quarter_urls[target_quarter]
            if target_quarter < len(self._quarter_urls)
            else None
        )
        if not target_url:
            return transaction

        if transaction is None:
            transaction = MigrationTransaction(
                role=self._role,
                session_id=session_id,
                source_url=self._full_url,
                target_quarter=target_quarter,
                target_url=target_url,
            )

        slot_filename = _slot_filename(self._role, session_id, transaction.txn_id)

        # The file is written by the SOURCE server and read by the TARGET's
        # restore; `_slot_erase` below clears only in-memory state, so the file
        # is removed here once no restore can still read it (see
        # _slot_file_safe_to_unlink) — on every exit, including exceptions.
        self._slot_file_begin(slot_filename)
        try:
            transaction.advance(MigrationState.SAVING)
            saved = _slot_save(self._full_url, filename=slot_filename)
            n_saved = _reported_tokens(saved)
            # `not saved` now also catches a 0-token save, which is an HTTP 200 and
            # was previously indistinguishable from a real one. Nine such 752-byte
            # artifacts sit in the slot cache; four share a name class with 64 real
            # saves, so this is a failure mode of the normal path, not of probes.
            if not saved:
                transaction.advance(MigrationState.ABORTED, detail="save_failed")
                # State-only update, NOT _finalize_quarter_assignment: an ABORTED
                # migration must not write the HARD affinity map (`_session_quarter`),
                # which is the SUCCESS path's commit record. The asymmetry is safe
                # because `_quarter_for_session_locked` falls back to the session
                # STATE and already treats _STATE_MIGRATION_FAILED_COLD as a reserved
                # quarter assignment — so dispatch still pins this session to
                # `target_quarter` while nothing downstream mistakes a failed
                # handover for a committed one.
                with self._lock:
                    self._set_session_state(
                        session_id,
                        state=_STATE_MIGRATION_FAILED_COLD,
                        quarter=target_quarter,
                        detail="save_failed",
                    )
                    self._migration_failures += 1
                logger.warning(
                    "KV migration save failed for %s session=%s, quarter %d starts cold (txn=%s)",
                    self._role, session_id, target_quarter, transaction.txn_id,
                )
                return transaction

            transaction.advance(MigrationState.RESTORING)
            restored = _slot_restore(target_url, filename=slot_filename)
            n_restored = _reported_tokens(restored)
            if not restored:
                transaction.advance(MigrationState.ABORTED, detail="restore_failed")
                # See the save_failed branch above: state-only, no hard affinity.
                with self._lock:
                    self._set_session_state(
                        session_id,
                        state=_STATE_MIGRATION_FAILED_COLD,
                        quarter=target_quarter,
                        detail="restore_failed",
                    )
                    self._migration_failures += 1
                logger.warning(
                    "KV migration restore failed for %s session=%s quarter %d; session will run cold (txn=%s)",
                    self._role, session_id, target_quarter, transaction.txn_id,
                )
                return transaction

            # VERIFIED must mean "the KV came back", not "the request was served".
            # HTTP 200 proves transport; n_restored proves reuse. Until 2026-08-22
            # this advanced on the status code alone and the NEXT statement erased
            # the source, so a zero-token restore destroyed the only good copy and
            # recorded it as a success. The server already returns the number and
            # _slot_restore already parsed it; nothing read it.
            if n_saved is not None and n_restored is not None and n_restored != n_saved:
                transaction.advance(MigrationState.ABORTED, detail="restore_token_mismatch")
                with self._lock:
                    self._set_session_state(
                        session_id,
                        state=_STATE_MIGRATION_FAILED_COLD,
                        quarter=target_quarter,
                        detail="restore_token_mismatch",
                    )
                    self._migration_failures += 1
                logger.error(
                    "KV migration restore returned %d tokens but %d were saved for %s "
                    "session=%s quarter %d; SOURCE SLOT PRESERVED and session runs cold (txn=%s)",
                    n_restored, n_saved, self._role, session_id, target_quarter,
                    transaction.txn_id,
                )
                return transaction

            # Restore confirmed — placement waiters may now proceed (audit refinement:
            # incoming request must wait for VERIFIED before placing on the freed slot).
            transaction.advance(MigrationState.VERIFIED, detail="restore_confirmed")

            # Source erase happens AFTER verification — destructive on failure, so
            # we want to be 100% sure the restore succeeded before clearing source.
            # That certainty now comes from the token-count equality gate above.
            _slot_erase(self._full_url)
            transaction.advance(MigrationState.SOURCE_ERASED)

            self._finalize_quarter_assignment(
                session_id,
                target_quarter,
                state=_STATE_ASSIGNED_QUARTER,
                detail="restored",
            )
            transaction.advance(MigrationState.COMMITTED)

            logger.info(
                "KV migration complete: %s session=%s full → quarter %d (%.0fms, txn=%s)",
                self._role, session_id, target_quarter,
                transaction.elapsed_ms, transaction.txn_id,
            )
            return transaction
        finally:
            self._slot_file_end(slot_filename, transaction)

    def _release(self, idx: int, is_full: bool) -> None:
        with self._lock:
            if is_full:
                self._full_active = False
                self._full_idle_since = time.monotonic()
            elif 0 <= idx < len(self._quarter_active):
                self._quarter_active[idx] = False
        # WP-4: opportunistically migrate the released quarter's session back
        # to full when load drops, cooldown elapsed, session is warm, and the
        # per-session migration cap hasn't been exceeded. Spawned as a daemon
        # thread so the dispatcher's finally-block stays fast.
        if (
            not is_full
            and self._reverse_migration_enabled()
            # INF-78 OAB-3 (R2): quiescent_after → no reverse-migration thread at request end.
            and not _quiescence_suppress("kv_reverse_migration")
        ):
            self._maybe_spawn_reverse_migration(idx)

    @staticmethod
    def _reverse_migration_enabled() -> bool:
        """WP-4: gate the quarter→full reverse migration trigger. Default off
        until the gate passes (30-min mixed traffic shows reverse migrations
        firing; solo-after-burst latency ≤+10% vs solo-only baseline)."""
        import os as _os
        return _os.environ.get("ORCHESTRATOR_REVERSE_MIGRATION", "0").strip() in {
            "1", "true", "yes", "on",
        }

    def _maybe_spawn_reverse_migration(self, released_quarter_idx: int) -> None:
        """WP-4: check the four reverse-migration guards and, if all pass,
        spawn `_reverse_migrate_kv(session_id, source_quarter)` in a daemon
        thread. Idempotent — all guards short-circuit on failure.
        """
        now = time.monotonic()
        # Find the session whose quarter affinity matches the released quarter.
        with self._lock:
            session_id: Optional[str] = None
            for sid, q_idx in self._session_quarter.items():
                if q_idx == released_quarter_idx:
                    session_id = sid
                    break
            if session_id is None:
                return  # released quarter has no affinity owner

            # Guard 1: full has been idle for ≥ cooldown.
            cooldown_s = self._reverse_migration_cooldown_ms() / 1000.0
            if self._full_active or self._full_idle_since is None:
                return
            if (now - self._full_idle_since) < cooldown_s:
                # Anti-thrash guard ("avoids thrashing", handoff Phase 4): full
                # released too recently to warm a session back onto it.
                _record_thrash_skip("cooldown")
                return

            # Guard 2: session has had a recent request (within window).
            window_s = self._reverse_migration_window_ms() / 1000.0
            last_seen = self._session_last_seen.get(session_id, 0.0)
            if last_seen == 0.0 or (now - last_seen) > window_s:
                return  # session idle too long to be worth warming back to full

            # Guard 3: per-session migration cap (avoid ping-pong).
            cap = self._reverse_migration_session_cap()
            if self._reverse_migration_counts.get(session_id, 0) >= cap:
                # Anti-thrash guard ("avoids ping-pong", handoff Phase 4): this
                # session already hit its per-session reverse-migration cap.
                _record_thrash_skip("session_cap")
                return

            # Guard 4: don't double-fire.
            if self._reverse_migration_in_flight.get(session_id, False):
                return
            self._reverse_migration_in_flight[session_id] = True
            self._reverse_migration_counts[session_id] = (
                self._reverse_migration_counts.get(session_id, 0) + 1
            )

        threading.Thread(
            target=self._reverse_migrate_kv,
            args=(session_id, released_quarter_idx),
            daemon=True,
            name=f"kv-reverse-{self._role}-{session_id[:8]}",
        ).start()

    @staticmethod
    def _reverse_migration_cooldown_ms() -> int:
        import os as _os
        try:
            return max(0, int(_os.environ.get("ORCHESTRATOR_REVERSE_MIGRATION_COOLDOWN_MS", "2000")))
        except (TypeError, ValueError):
            return 2000

    @staticmethod
    def _reverse_migration_window_ms() -> int:
        import os as _os
        try:
            return max(0, int(_os.environ.get("ORCHESTRATOR_REVERSE_MIGRATION_WINDOW_MS", "30000")))
        except (TypeError, ValueError):
            return 30000

    @staticmethod
    def _reverse_migration_session_cap() -> int:
        import os as _os
        try:
            return max(0, int(_os.environ.get("ORCHESTRATOR_REVERSE_MIGRATION_SESSION_CAP", "5")))
        except (TypeError, ValueError):
            return 5

    def _reverse_migrate_kv(self, session_id: str, source_quarter: int) -> None:
        """WP-4: quarter → full migration. Mirrors _migrate_kv but in the
        opposite direction. Uses a MigrationTransaction so observability
        is consistent across forward and reverse. Best-effort — on failure
        the session stays on the quarter and the in-flight flag is cleared
        so a future opportunity can retry."""
        from src.scheduling.migration_transaction import (
            MigrationState,
            MigrationTransaction,
        )

        slot_filename: str | None = None
        txn: Any = None
        try:
            if not self._full_url:
                return
            source_url = (
                self._quarter_urls[source_quarter]
                if source_quarter < len(self._quarter_urls)
                else None
            )
            if not source_url:
                return

            txn = MigrationTransaction(
                role=self._role,
                session_id=session_id,
                source_url=source_url,
                target_quarter=-1,  # convention: -1 = full
                target_url=self._full_url,
            )
            slot_filename = _slot_filename(self._role, session_id, txn.txn_id)
            self._slot_file_begin(slot_filename)

            txn.advance(MigrationState.SAVING)
            rev_saved = _slot_save(source_url, filename=slot_filename)
            rev_n_saved = _reported_tokens(rev_saved)
            if not rev_saved:
                txn.advance(MigrationState.ABORTED, detail="save_failed")
                logger.warning(
                    "WP-4 reverse migration save failed role=%s session=%s txn=%s",
                    self._role, session_id, txn.txn_id,
                )
                return

            txn.advance(MigrationState.RESTORING)
            rev_restored = _slot_restore(self._full_url, filename=slot_filename)
            rev_n_restored = _reported_tokens(rev_restored)
            if not rev_restored:
                txn.advance(MigrationState.ABORTED, detail="restore_failed")
                logger.warning(
                    "WP-4 reverse migration restore failed role=%s session=%s txn=%s",
                    self._role, session_id, txn.txn_id,
                )
                return

            # Same equality gate as the forward path: the erase below is
            # destructive and HTTP 200 is not evidence the KV came back.
            if (
                rev_n_saved is not None
                and rev_n_restored is not None
                and rev_n_restored != rev_n_saved
            ):
                txn.advance(MigrationState.ABORTED, detail="restore_token_mismatch")
                logger.error(
                    "WP-4 reverse migration restored %d tokens but %d were saved "
                    "role=%s session=%s txn=%s; SOURCE SLOT PRESERVED",
                    rev_n_restored, rev_n_saved, self._role, session_id, txn.txn_id,
                )
                return

            txn.advance(MigrationState.VERIFIED, detail="restore_confirmed")
            _slot_erase(source_url)
            txn.advance(MigrationState.SOURCE_ERASED)

            # Clear the quarter affinity — session now belongs to full again.
            with self._lock:
                if self._session_quarter.get(session_id) == source_quarter:
                    self._session_quarter.pop(session_id, None)
                self._full_last_session = session_id
                self._set_session_state(
                    session_id,
                    state=_STATE_ASSIGNED_FULL,
                    quarter=None,
                    detail="reverse_migrated",
                )
            txn.advance(MigrationState.COMMITTED)
            logger.info(
                "WP-4 reverse migration committed role=%s session=%s quarter%d→full (%.0fms, txn=%s)",
                self._role, session_id, source_quarter, txn.elapsed_ms, txn.txn_id,
            )
        finally:
            if slot_filename is not None:
                self._slot_file_end(slot_filename, txn)
            with self._lock:
                self._reverse_migration_in_flight.pop(session_id, None)

    def clear_session(self, session_id: str) -> None:
        """Remove session affinity (call when session completes)."""
        with self._lock:
            self._session_quarter.pop(session_id, None)
            self._session_state.pop(session_id, None)
            if self._full_last_session == session_id:
                self._full_last_session = None

    # === Forward all backend interface methods ===
    # Note: session_id extraction from request is best-effort.
    # If request has no session_id, routing falls back to load-based selection.

    def _extract_session_id(self, request: Any) -> str:
        """Try to extract session_id from request for affinity routing."""
        if hasattr(request, "session_id"):
            return str(request.session_id or "")
        if hasattr(request, "task_id"):
            return str(request.task_id or "")
        return ""

    # === Cross-process dispatch (Phase 2 of per-region locks) ===
    #
    # When ORCHESTRATOR_PER_REGION_LOCKS=1, the dispatch path uses
    # `cpu_region_lock_for_instance` non-blocking try-acquire to pick an
    # instance that's actually free across uvicorn worker processes. The
    # legacy in-process `_full_active` / `_quarter_active` flags are still
    # maintained for telemetry but are no longer the authoritative
    # cross-process source of truth. When the flag is off, legacy
    # `_select` is used unchanged.

    @staticmethod
    def _per_region_locks_enabled() -> bool:
        import os as _os
        return _os.environ.get("ORCHESTRATOR_PER_REGION_LOCKS", "0").strip() in {
            "1", "true", "yes", "on",
        }

    @staticmethod
    def _placement_state_machine_enabled() -> bool:
        """WP-2: gate the topology-safe placement filter + poll-on-queue
        fallback. Default off — operators flip to 1 after the WP-2 gate
        (4-way frontdoor shows 3 active + 1 queued, never overlap).

        When off, falls back to the prior greedy try-loop + blocking-on-full
        fallback so existing deployments are unaffected by this code landing.
        """
        import os as _os
        return _os.environ.get("ORCHESTRATOR_PLACEMENT_STATE_MACHINE", "0").strip() in {
            "1", "true", "yes", "on",
        }

    @staticmethod
    def _cross_role_disjoint_placement_enabled() -> bool:
        """Part A (shape-keyed-contention-gating): when on, placement excludes
        regions held by OTHER roles too, so a light role can backfill the free
        quarters beside a heavy role's node-half. Default off — flips behavior
        only after the live region-lock observation gate is met. Effective only
        when ORCHESTRATOR_PLACEMENT_STATE_MACHINE=1 (which gates the call site)."""
        import os as _os
        return _os.environ.get("ORCHESTRATOR_CROSS_ROLE_DISJOINT_PLACEMENT", "0").strip() in {
            "1", "true", "yes", "on",
        }

    @staticmethod
    def _shape_aware_contention_enabled() -> bool:
        """True when the B seam should evaluate real dispatch candidates."""
        try:
            from src.scheduling.contention import shape_aware_contention_enabled

            return shape_aware_contention_enabled()
        except Exception:
            return False

    @staticmethod
    def _traffic_class_for_request(request: Any):
        from src.scheduling.contention import TrafficClass

        priority = getattr(request, "request_priority", "interactive")
        if str(priority).strip().lower() == "background":
            return TrafficClass.BACKGROUND
        return TrafficClass.FOREGROUND_INTERACTIVE

    @staticmethod
    def _max_queue_wait_ms_for_request(request: Any) -> int | None:
        raw = getattr(request, "max_queue_wait_ms", None)
        if raw is None:
            return None
        try:
            value = int(raw)
        except (TypeError, ValueError):
            return None
        return value if value >= 0 else None

    def _compute_full_slot_alignment(self) -> bool:
        """DISPATCH-A: True unless the endpoint wired as this backend's full
        instance is a mislabeled quarter (its port != the topology role's
        NUMA_CONFIG idx-0 port). Unknown expected port → True (legacy).

        No full backend (quarters-only, e.g. after a misaligned full is demoted
        upstream) → True: there is no full slot to be misaligned. `emit_full`
        independently requires `self._full is not None`, so this never emits a
        phantom full."""
        if self._full is None:
            return True
        try:
            from src.runtime.instance_topology import full_instance_port

            expected = full_instance_port(self._topology_role)
        except Exception:
            return True
        if expected is None:
            return True
        return self._full_port == expected

    def _extract_migration_budget_ms(self, request: Any) -> Optional[int]:
        """Best-effort read of ChatRequest.migration_budget_ms for the WP-3
        load-transition trigger. Returns None if not present."""
        if request is None:
            return None
        v = getattr(request, "migration_budget_ms", None)
        if v is None:
            return None
        try:
            iv = int(v)
            return iv if iv > 0 else None
        except (TypeError, ValueError):
            return None

    # ── WP-12 fleet layer: per-endpoint health (one circuit per fleet port) ──

    @property
    def fleet_health_managed(self) -> bool:
        """True when this backend records per-endpoint health itself (WP-12
        one-CAB-per-fleet construction). The primitives layer then skips its
        legacy primary-URL health bookkeeping for calls through this backend."""
        return self._health_tracker is not None

    def endpoint_urls(self) -> list[str]:
        """Every live endpoint URL of this backend (full first when present)."""
        urls = [u for u in self._quarter_urls if u]
        if self._full_url:
            urls.insert(0, self._full_url)
        return urls

    def any_endpoint_available(self) -> bool:
        """False iff the FLEET circuit is open (every endpoint circuit open)."""
        if self._health_tracker is None:
            return True
        urls = self.endpoint_urls()
        if not urls:
            return True
        return any(self._health_tracker.is_available(u) for u in urls)

    def _endpoint_url_for_idx(self, idx: int) -> str | None:
        if idx == -1:
            return self._full_url
        if 0 <= idx < len(self._quarter_urls):
            return self._quarter_urls[idx]
        return None

    def _candidate_available(self, ca_idx: int) -> bool:
        if self._health_tracker is None:
            return True
        url = self._endpoint_url_for_idx(ca_idx)
        if not url:
            return True
        return self._health_tracker.is_available(url)

    def _record_endpoint_result(self, idx: int, result: Any) -> None:
        """Mirror the primitives layer's health policy per dispatched endpoint:
        partial results are neither success nor failure."""
        if self._health_tracker is None:
            return
        url = self._endpoint_url_for_idx(idx)
        if not url:
            return
        if isinstance(getattr(result, "context_overflow", None), dict):
            # Request-sized (or load-induced) KV exhaustion is not endpoint
            # ill-health; counting it would open the circuit on a healthy server.
            return
        success = bool(getattr(result, "success", True))
        partial = bool(getattr(result, "partial", False))
        if success and not partial:
            self._health_tracker.record_success(url)
        elif not success and not partial:
            self._health_tracker.record_failure(url)

    def _tap_dispatch_metadata(
        self, idx: int, backend: Any, logical_role: str | None = None
    ) -> dict[str, Any]:
        """Return structured tap metadata for the selected runtime instance.

        ``logical_role`` threads the per-call role through a fleet-shared
        backend (WP-12): the tap's ``role`` stays the LOGICAL role while
        ``topology_role``/``lock_role`` remain the physical fleet identity.
        Legacy per-role backends pass None (or the identical role string).
        """
        topology_idx = 0 if idx == -1 else self._quarter_topology_idx[idx]
        url = self._full_url if idx == -1 else (
            self._quarter_urls[idx] if 0 <= idx < len(self._quarter_urls) else None
        )
        port = _extract_port(url) or (self._full_port if idx == -1 else None)
        regions: frozenset[str] = frozenset()
        try:
            from src.runtime.instance_topology import get_instance_regions

            regions = get_instance_regions().get((self._topology_role, topology_idx), frozenset())
        except Exception:
            regions = frozenset()
        return {
            "role": logical_role or self._role,
            "topology_role": self._topology_role,
            "lock_role": self._topology_role,
            "instance_idx": topology_idx,
            "concurrency_idx": idx,
            "instance_shape": _shape_for_regions(regions),
            "instance_regions": sorted(regions),
            "port": port,
            "backend_url": url or _get_base_url(backend),
        }

    def _annotate_current_tap_dispatch(
        self,
        idx: int,
        backend: Any,
        logical_role: str | None = None,
        request: Any = None,
    ) -> None:
        metadata = self._tap_dispatch_metadata(idx, backend, logical_role=logical_role)
        try:
            from src.inference_tap import annotate_current_tap

            annotate_current_tap(**metadata)
        except Exception:
            pass
        try:
            from src.runtime.live_telemetry import emit_lifecycle_transition

            emit_lifecycle_transition(
                "backend_dispatched",
                request_id=getattr(request, "inference_request_id", None),
                task_id=getattr(request, "task_id", None),
                batch_id=getattr(request, "batch_id", None),
                role=metadata.get("role"),
                port=metadata.get("port"),
                details={
                    "backend_url": metadata.get("backend_url"),
                    "topology_role": metadata.get("topology_role"),
                    "instance_idx": metadata.get("instance_idx"),
                    "instance_shape": metadata.get("instance_shape"),
                    "batch_placement_mode": getattr(
                        request, "batch_placement_mode", "auto"
                    ),
                },
            )
        except Exception:
            pass

    @contextmanager
    def _dispatch(
        self,
        session_id: str = "",
        migration_budget_ms: Optional[int] = None,
        request: Any = None,
    ):
        """Yield (backend, idx, is_full) with the right lock held.

        Two paths:
        - Legacy (default): pre-existing in-process selection via _select.
        - Per-region (flag on): cross-process region locks coordinate
          which instance handles the request.

        On context exit, releases both in-process state and any region
        locks. Caller calls backend.infer/infer_stream_text inside the
        with-block; nothing else.
        """
        # WP-4: stamp per-session last-seen so the reverse-migration trigger
        # can distinguish warm sessions (recently active → worth migrating
        # back to full) from cold ones (idle long enough that the KV save
        # cost isn't repaid by future requests).
        if session_id:
            with self._lock:
                self._session_last_seen[session_id] = time.monotonic()
        if not self._per_region_locks_enabled():
            backend, idx, is_full = self._select(session_id=session_id)
            try:
                yield (backend, idx, is_full)
            finally:
                self._release(idx, is_full)
            return

        # Per-region cross-process path. Try instances in priority order:
        #  1. full (best per-request latency, warm KV)
        #  2. each quarter, in numerical order
        # First non-blocking acquire that succeeds wins. If all fail,
        # fall through to blocking acquire on full (mirrors the legacy
        # "all-busy → overflow to full" behavior, but waits properly).
        from src.runtime.cpu_region_lock import (
            cpu_region_lock_for_instance,
            CpuRegionLockTimeout,
            _cross_role_mutex_enabled,
        )
        from src.scheduling.placement_policy import (
            BatchPlacementMode,
            RolePlacementPolicy,
            coerce_batch_placement_mode,
            get_placement_policy,
        )

        placement_policy = get_placement_policy(self._topology_role)
        batch_placement_mode = coerce_batch_placement_mode(
            getattr(request, "batch_placement_mode", None)
        )

        # Certified EvalTower requests share one placement lease on the same
        # full llama-server process.  LOCK_SH lets its native slots co-run;
        # every other CPU placement still takes LOCK_EX and remains mutually
        # exclusive with this lane.  Requiring both workload_class and batch_id
        # keeps ordinary/background clients off the shared path.
        native_batch = (
            self._native_batch_width > 1
            and _cross_role_mutex_enabled()
            and self._full is not None
            and self._full_slot_aligned
            and str(getattr(request, "workload_class", "") or "") == "eval_batch"
            and bool(str(getattr(request, "batch_id", "") or "").strip())
            and batch_placement_mode is not BatchPlacementMode.MIXED_ROLE_SPLIT
        )
        if native_batch:
            request_tag = str(getattr(request, "batch_id", "") or "")
            try:
                placement_timeout_s = max(
                    60.0, float(getattr(request, "timeout", 60.0) or 60.0)
                )
            except (TypeError, ValueError):
                placement_timeout_s = 60.0
            shared_ctx = cpu_region_lock_for_instance(
                self._topology_role,
                0,
                timeout_s=placement_timeout_s,
                request_tag=request_tag,
                shared=True,
                capacity=self._native_batch_width,
            )
            shared_ctx.__enter__()
            with self._lock:
                self._total_requests += 1
                self._full_requests += 1
                self._native_full_active += 1
                self._full_active = True
            try:
                yield (self._full, -1, True)
            finally:
                with self._lock:
                    self._native_full_active = max(0, self._native_full_active - 1)
                    self._full_active = self._native_full_active > 0
                    if not self._full_active:
                        self._full_idle_since = time.monotonic()
                shared_ctx.__exit__(None, None, None)
            return

        deadline_s = None  # caller's deadline carried in via request.timeout

        def _try_instance(role: str, instance_idx: int):
            """Try non-blocking acquire of the region locks for one
            instance. Returns an entered context-manager handle on
            success, or None if the lock is held elsewhere."""
            ctx = cpu_region_lock_for_instance(
                role, instance_idx,
                timeout_s=0.05,  # effectively non-blocking
                deadline_s=None,
            )
            try:
                paths = ctx.__enter__()
            except CpuRegionLockTimeout:
                return None
            return ctx, paths

        # Priority list of (idx_in_concurrency_aware, instance_idx_in_topology)
        # In NUMA_CONFIG: idx 0 is full, idx 1..N are quarters → matches
        # our internal indexing (-1 for full, 0..N-1 for quarters).
        # Phase D: quarters are visited in `_quarter_preference_order` so when
        # the full lock is held, the next attempted quarter is one that's
        # NUMA-disjoint from full (no cpu-set overlap). For frontdoor full on
        # NUMA_NODE0 this picks q3/q2 before q1/q0.
        # 2026-05-24 Phase E port: when session_id has a known quarter
        # affinity from a prior migration (in `_session_quarter`), try THAT
        # quarter first so the warm KV state actually gets reused.
        sticky_q_idx: int | None = None
        if session_id:
            with self._lock:
                sticky_q_idx = self._quarter_for_session_locked(session_id)
        # DISPATCH-A: the placement policy governs whether/where the full
        # (-1, 0) all-region candidate is emitted.
        #   * FULL_DISABLED    — never emit full (quarters only).
        #   * BURST_PREFER_SPLIT — full FIRST only in single-request mode
        #     (zero current holders for the role); once any holder is present,
        #     quarters are preferred and the full trails. Because full="0-95"
        #     overlaps every quarter, a trailing full never places while a
        #     quarter holds a region — which is the mode-abandonment the design
        #     contract requires (full/half is single-request-throughput only).
        #   * SOLO_PREFER_FULL (default) / QUEUE_ONLY / unknown — legacy order
        #     (full first, then quarters).
        # The alignment guard (`_full_slot_aligned`) independently suppresses
        # the full candidate when the full slot is a mislabeled quarter.
        emit_full = (
            self._full is not None
            and self._full_slot_aligned
            and placement_policy is not RolePlacementPolicy.FULL_DISABLED
            and batch_placement_mode is not BatchPlacementMode.MIXED_ROLE_SPLIT
        )
        full_candidate = (-1, 0)

        # Base (single-request / legacy) ordering: sticky quarter (if any),
        # then full, then quarters in NUMA-disjoint preference order. `emit_full`
        # drops the full candidate entirely for FULL_DISABLED roles and for a
        # misaligned full slot. BURST_PREFER_SPLIT re-orders this list to
        # quarters-first inside the WP-2 poll loop below, but only once the role
        # has an in-flight holder (single-request mode keeps full first).
        candidates: list[tuple[int, int]] = []
        if sticky_q_idx is not None and 0 <= sticky_q_idx < len(self._quarters):
            candidates.append((sticky_q_idx, self._quarter_topology_idx[sticky_q_idx]))
        if emit_full:
            candidates.append(full_candidate)
        for q_idx in self._quarter_preference_order:
            if q_idx == sticky_q_idx:
                continue  # already at the head
            # internal quarter q_idx → its TRUE NUMA_CONFIG topology idx (port-resolved)
            candidates.append((q_idx, self._quarter_topology_idx[q_idx]))

        chosen_ctx = None
        chosen_idx = -2

        if self._placement_state_machine_enabled():
            # WP-2 path: topology-safe filter + poll-on-queue. The placement
            # policy filters dispatcher-priority `candidates` down to those
            # whose cpuset is disjoint from currently-held regions for this
            # role. If at least one survives, try-acquire each non-blocking.
            # If none survive (every candidate overlaps), poll until a release
            # makes a candidate safe; cap by 60s wall-clock (the same generous
            # bound the legacy block-on-full fallback used).
            #
            # WP-3 layer (this dispatch path is unchanged from WP-2): the
            # transactional migration model + policy gating + migration
            # budget threading apply to the EXISTING session-handover
            # trigger below (lines following "chosen_idx == -1"), not here.
            # A proactive load-transition trigger inside the poll loop was
            # explored and removed: _migrate_kv cannot preempt an in-flight
            # decode, so triggering it during a queue-wait does not unblock
            # the queue — the existing inference must complete and release
            # its lock, after which the WP-2 poll re-evaluates and succeeds.
            from src.scheduling.placement import evaluate_placement
            from src.runtime.cpu_region_lock import (
                active_region_holders,
                held_regions_by_role,
            )
            from src.runtime.instance_topology import get_instance_regions
            from src.scheduling.contention_gate import ContentionDenied, get_gate

            instance_regions = get_instance_regions()
            queue_wait_ms = self._max_queue_wait_ms_for_request(request)
            poll_budget_s = 60.0 if queue_wait_ms is None else max(0.0, queue_wait_ms / 1000.0)
            wait_budget_ms = int(poll_budget_s * 1000)
            workload_class = str(
                getattr(request, "workload_class", None) or "interactive"
            )
            poll_deadline = time.perf_counter() + poll_budget_s
            poll_interval_s = 0.150  # matches contention_gate._GATE_POLL_S
            queue_log_emitted = False
            cross_role_enabled = self._cross_role_disjoint_placement_enabled()
            shape_gate_enabled = self._shape_aware_contention_enabled()
            gate = get_gate() if shape_gate_enabled else None
            traffic_class = self._traffic_class_for_request(request)
            timeout_class = "admission_denied"
            timeout_code = "placement_unavailable"
            while True:
                all_holders = active_region_holders()
                holders_for_role = all_holders.get(self._topology_role, [])
                # DISPATCH-A: BURST_PREFER_SPLIT abandons the full instance for
                # split instances the moment the role has an in-flight holder (design
                # contract: full is single-request-throughput ONLY; under
                # concurrent load the router prefers quarters and lets the full
                # trail). Single-request (no self-role holder) keeps full first
                # for peak latency. Re-evaluated every poll so a request that
                # becomes concurrent mid-wait switches modes. Full="0-95"
                # overlaps every quarter, so a trailing full is naturally vetoed
                # while any quarter holds a region — no explicit eviction needed.
                loop_candidates = candidates
                if (
                    (
                        placement_policy is RolePlacementPolicy.BURST_PREFER_SPLIT
                        or batch_placement_mode is BatchPlacementMode.MIXED_ROLE_SPLIT
                    )
                    and emit_full
                    and holders_for_role
                ):
                    loop_candidates = [
                        c for c in candidates if c[0] != -1
                    ] + [full_candidate]
                # WP-12 fleet layer: skip endpoints whose circuit is open.
                # Health is one fact per (fleet, endpoint); a candidate whose
                # circuit is open cannot serve ANY bound role. When every
                # endpoint circuit is open the fleet is down — fail fast so
                # the role layer consults cross-fleet fallback instead of
                # burning the poll budget on dead ports. is_available()
                # itself admits the single half-open probe after cooldown
                # (fleet-global, not per bound role). No-op when no
                # health_tracker is wired (legacy per-role construction).
                if self._health_tracker is not None:
                    available = [
                        c for c in loop_candidates if self._candidate_available(c[0])
                    ]
                    if not available:
                        raise RuntimeError(
                            "Backend unavailable (circuit open): all endpoints "
                            f"for fleet {self._topology_role}"
                        )
                    loop_candidates = available
                # DISPATCH-A residual fix: feed the placement filter the EXACT
                # held-region sets, NOT the ATTRIBUTION idx view. `active_region_
                # holders()` marks an instance active when ANY of its regions is
                # held, so one held quarter (e.g. q0) is reported as every
                # instance containing q0 — including the all-region `full`
                # (idx 0). Expanding those idxs inflated the same-role holders_
                # union to the whole machine and QUEUED every physically-disjoint
                # quarter, serializing concurrent same-role traffic onto a single
                # quarter. `held_regions_by_role()` is the exact physical truth
                # (never over-reports the phantom full) and matches the
                # shape-aware seam. Recomputed per poll so releases are observed.
                held_by_role = held_regions_by_role(instance_regions)
                same_role_regions = held_by_role.get(
                    self._topology_role, frozenset()
                )
                cross_role_regions = None
                if cross_role_enabled:
                    _cross_acc: set[str] = set()
                    for _hrole, _hregions in held_by_role.items():
                        if _hrole != self._topology_role:
                            _cross_acc |= _hregions
                    cross_role_regions = frozenset(_cross_acc)
                placement = evaluate_placement(
                    role=self._topology_role,
                    candidates=loop_candidates,
                    holder_idxs=holders_for_role,
                    instance_regions=instance_regions,
                    cross_role_holders=all_holders if cross_role_enabled else None,
                    holder_regions=same_role_regions,
                    cross_role_regions=cross_role_regions,
                )
                if not placement.is_queue:
                    # At least one safe candidate — try them in priority order.
                    gate_rejected = False
                    lock_race_lost = False
                    for place in placement.places:
                        if gate is not None:
                            decision = gate.evaluate(
                                self._topology_role,
                                traffic_class,
                                candidate_topology_idx=place.topology_idx,
                            )
                            # BRIDGE RESIDUAL 1 — this is the per-INSTANCE verdict
                            # the role-granular proxy could never see. Recorded for
                            # every candidate tried, admitted or not, so the probe
                            # observes the walk down the priority order rather than
                            # only its outcome. Observational; the branch below is
                            # unchanged.
                            gate_observation.record(
                                admitted=decision.admitted,
                                decision=getattr(
                                    decision.decision, "value", str(decision.decision)
                                ),
                                waited_s=decision.waited_s,
                                candidate_topology_idx=place.topology_idx,
                                blocking_roles=list(decision.blocking_roles or []),
                                reason=decision.reason,
                                role=self._topology_role,
                            )
                            if not decision.admitted:
                                if not queue_log_emitted:
                                    logger.info(
                                        "placement-aware contention queued role=%s "
                                        "topology_idx=%d decision=%s reason=%s",
                                        self._role,
                                        place.topology_idx,
                                        decision.decision.value,
                                        decision.reason,
                                    )
                                    queue_log_emitted = True
                                gate_rejected = True
                                continue
                        attempt = _try_instance(self._topology_role, place.topology_idx)
                        if attempt is not None:
                            chosen_ctx, _paths = attempt
                            chosen_idx = place.internal_idx
                            break
                        lock_race_lost = True
                    if chosen_ctx is not None:
                        break  # acquired; exit poll loop
                    # Either every physically safe candidate was rejected by
                    # the placement-aware seam, or every safe candidate's lock
                    # was stolen between the holder snapshot and try-acquire.
                    # Re-poll until holders change or the budget expires.
                    if gate_rejected:
                        timeout_class = "admission_denied"
                        timeout_code = "placement_gate_timeout"
                    elif lock_race_lost:
                        # A safe candidate existed, but another request acquired
                        # it between the holder snapshot and our try-acquire.
                        timeout_class = "admission_timeout"
                        timeout_code = "race_lost"
                    else:
                        timeout_class = "admission_denied"
                        timeout_code = "placement_unavailable"
                else:
                    if not queue_log_emitted:
                        logger.info(
                            "placement queued role=%s reason=%s detail=%s",
                            self._role, placement.queue.reason.value, placement.queue.detail,
                        )
                        queue_log_emitted = True
                    timeout_class = "admission_denied"
                    timeout_code = f"placement_{placement.queue.reason.value}_timeout"
                if time.perf_counter() >= poll_deadline:
                    raise ContentionDenied(
                        f"placement timeout role={self._role} "
                        f"reason={timeout_code} holders={holders_for_role} after {poll_budget_s:.1f}s",
                        role=self._topology_role,
                        workload_class=workload_class,
                        wait_budget_ms=wait_budget_ms,
                        failure_class=timeout_class,
                        code=timeout_code,
                    )
                time.sleep(poll_interval_s)
        else:
            # Legacy path: greedy try-loop + blocking-on-full fallback.
            for ca_idx, topo_idx in candidates:
                attempt = _try_instance(self._topology_role, topo_idx)
                if attempt is not None:
                    chosen_ctx, _paths = attempt
                    chosen_idx = ca_idx
                    break

            if chosen_ctx is None:
                if emit_full:
                    # All non-blocking attempts failed → block on full's region
                    # locks. Lock layer's union-acquisition prevents overlap
                    # (full's lock takes all of full's regions), but this can
                    # wait on full even when a quarter would have been safe sooner.
                    blocking_ctx = cpu_region_lock_for_instance(
                        self._topology_role, 0,
                        timeout_s=60.0, deadline_s=deadline_s,
                    )
                    chosen_ctx = blocking_ctx
                    blocking_ctx.__enter__()
                    chosen_idx = -1
                else:
                    # Quarters-only (no full slot, e.g. a demoted misaligned
                    # full): there is no all-region idx-0 instance to block on.
                    # Block on the first-preference quarter's TRUE region locks
                    # instead — never grab the unserved all-region idx-0 lock.
                    first_q = (
                        self._quarter_preference_order[0]
                        if self._quarter_preference_order
                        else 0
                    )
                    blocking_ctx = cpu_region_lock_for_instance(
                        self._topology_role, self._quarter_topology_idx[first_q],
                        timeout_s=60.0, deadline_s=deadline_s,
                    )
                    chosen_ctx = blocking_ctx
                    blocking_ctx.__enter__()
                    chosen_idx = first_q

        # Update in-process telemetry (best-effort; not authoritative).
        # 2026-05-24 Phase E port: when full is acquired by a DIFFERENT session
        # than the previous holder AND legacy migration is enabled (via
        # `_kv_migration_enabled` — defaults disabled under PER_REGION_LOCKS=1
        # but operators can re-enable per-instance for opt-in testing of the
        # ported migration path), kick off async save/restore of the OLD
        # session's KV to a disjoint quarter.
        migrate_old_session: str | None = None
        migrate_target_quarter: int | None = None
        # WP-3 policy gate: FULL_DISABLED + QUEUE_ONLY skip migration; the
        # other two policies (SOLO_PREFER_FULL = default, BURST_PREFER_SPLIT)
        # leave the existing session-handover migration trigger active.
        # Reuse the policy resolved during candidate construction (same scope).
        _migration_allowed_by_policy = placement_policy in (
            RolePlacementPolicy.SOLO_PREFER_FULL,
            RolePlacementPolicy.BURST_PREFER_SPLIT,
        )
        with self._lock:
            self._total_requests += 1
            if chosen_idx == -1:
                self._full_active = True
                self._full_requests += 1
                old_session = self._full_last_session
                if (
                    self._kv_migration_enabled
                    and _migration_allowed_by_policy
                    and old_session
                    and session_id
                    and old_session != session_id
                    and self._quarter_for_session_locked(old_session) is None
                ):
                    # Find an idle quarter in preference order (disjoint first)
                    for q_idx in self._quarter_preference_order:
                        if not self._quarter_active[q_idx]:
                            # INF-78 OAB-3 (R2): quiescent_after → no migration thread.
                            if _quiescence_suppress("kv_migration"):
                                break
                            migrate_old_session = old_session
                            migrate_target_quarter = q_idx
                            # Reserve the quarter so concurrent dispatch doesn't grab it
                            self._migrations += 1
                            self._set_session_state(
                                old_session,
                                state=_STATE_MIGRATION_PENDING,
                                quarter=q_idx,
                            )
                            break
                if session_id:
                    self._full_last_session = session_id
                    self._set_session_state(
                        session_id,
                        state=_STATE_ASSIGNED_FULL,
                        quarter=None,
                    )
            else:
                self._quarter_active[chosen_idx] = True
                self._quarter_requests += 1
                if session_id:
                    self._session_quarter[session_id] = chosen_idx

        # Kick off migration outside the lock (the thread does its own locking).
        # WP-3: pre-allocate the MigrationTransaction with the per-request
        # budget honored from ChatRequest.migration_budget_ms (default 30s
        # when no per-request override; matches the legacy _SLOT_SAVE_TIMEOUT).
        if migrate_old_session is not None and migrate_target_quarter is not None:
            from src.scheduling.migration_transaction import MigrationTransaction
            _target_url = (
                self._quarter_urls[migrate_target_quarter]
                if migrate_target_quarter < len(self._quarter_urls)
                else ""
            )
            _txn = MigrationTransaction(
                role=self._role,
                session_id=migrate_old_session,
                source_url=self._full_url or "",
                target_quarter=migrate_target_quarter,
                target_url=_target_url,
                migration_budget_ms=(migration_budget_ms or 30_000),
            )
            threading.Thread(
                target=self._migrate_kv,
                args=(migrate_old_session, migrate_target_quarter, _txn),
                daemon=True,
                name=f"kv-migrate-{self._role}-{migrate_old_session[:8]}-{_txn.txn_id}",
            ).start()

        # Telemetry: stamp quarter-assignment session state outside the lock
        # (orphaned from the with-self._lock block above during the 2026-05-24
        # migration-port refactor; quarter branch sets this here.)
        if chosen_idx != -1 and session_id:
            with self._lock:
                self._set_session_state(
                    session_id,
                    state=_STATE_ASSIGNED_QUARTER,
                    quarter=chosen_idx,
                )

        backend = self._full if chosen_idx == -1 else self._quarters[chosen_idx]
        is_full = (chosen_idx == -1)

        try:
            yield (backend, chosen_idx, is_full)
        finally:
            # In-process release
            self._release(chosen_idx, is_full)
            # Region-lock release
            try:
                chosen_ctx.__exit__(None, None, None)
            except Exception as exc:
                logger.warning(
                    "region-lock release failed (role=%s idx=%d): %s",
                    self._role, chosen_idx, exc,
                )

    def infer(self, role_config: Any, request: Any) -> Any:
        sid = self._extract_session_id(request)
        mb = self._extract_migration_budget_ms(request)
        logical_role = getattr(request, "role", None)
        _placement_t0 = time.perf_counter()
        with self._dispatch(session_id=sid, migration_budget_ms=mb, request=request) as (backend, _idx, _is_full):
            serving_calls.annotate_staged(
                instance_idx=_idx,
                instance_full=_is_full,
                placement_wait_ms=round((time.perf_counter() - _placement_t0) * 1000.0, 3),
            )
            self._annotate_current_tap_dispatch(
                _idx, backend, logical_role=logical_role, request=request
            )
            # Endpoint health is recorded from RESULT objects only (the
            # backend folds transport errors into success=False results;
            # raw exceptions are cancellation/lock control flow and must
            # not trip the circuit) — mirrors the primitives-layer policy.
            result = backend.infer(role_config, request)
            self._record_endpoint_result(_idx, result)
            return result

    def infer_streaming(self, role_config: Any, request: Any) -> Any:
        sid = self._extract_session_id(request)
        mb = self._extract_migration_budget_ms(request)
        logical_role = getattr(request, "role", None)
        with self._dispatch(session_id=sid, migration_budget_ms=mb, request=request) as (backend, _idx, _is_full):
            self._annotate_current_tap_dispatch(
                _idx, backend, logical_role=logical_role, request=request
            )
            # Streaming handle: outcome is not observable here — endpoint
            # health for this path stays with the consumer (as today).
            return backend.infer_streaming(role_config, request)

    def infer_stream_text(self, role_config: Any, request: Any, on_chunk: Any = None) -> Any:
        sid = self._extract_session_id(request)
        mb = self._extract_migration_budget_ms(request)
        logical_role = getattr(request, "role", None)
        _placement_t0 = time.perf_counter()
        with self._dispatch(session_id=sid, migration_budget_ms=mb, request=request) as (backend, _idx, _is_full):
            serving_calls.annotate_staged(
                instance_idx=_idx,
                instance_full=_is_full,
                placement_wait_ms=round((time.perf_counter() - _placement_t0) * 1000.0, 3),
            )
            self._annotate_current_tap_dispatch(
                _idx, backend, logical_role=logical_role, request=request
            )
            result = backend.infer_stream_text(role_config, request, on_chunk=on_chunk)
            self._record_endpoint_result(_idx, result)
            return result

    def health_check(self, pid: int = 0) -> bool:
        """Check health of full instance + all quarters."""
        full_ok = self._full.health_check(pid)
        quarters_ok = all(q.health_check(pid) for q in self._quarters)
        return full_ok and quarters_ok

    def get_stats(self) -> dict[str, Any]:
        """Telemetry for observability (DS-1 compatible)."""
        with self._lock:
            quarter_active = list(self._quarter_active)
            full_active = self._full_active
            session_map = dict(self._session_quarter)
            session_state = {
                sid: dict(state)
                for sid, state in self._session_state.items()
            }

        return {
            "role": self._role,
            "topology_role": self._topology_role,
            "backend_type": "concurrency_aware",
            "full_instance": {
                "port": self._full_port,
                "active": full_active,
                "total_served": self._full_requests,
                "current_session": self._full_last_session,
            },
            "quarter_instances": len(self._quarters),
            "quarter_active": quarter_active,
            "total_active": (1 if full_active else 0) + sum(quarter_active),
            "idle_quarters": sum(1 for a in quarter_active if not a),
            "total_requests": self._total_requests,
            "full_requests": self._full_requests,
            "quarter_requests": self._quarter_requests,
            "migrations": self._migrations,
            "migration_failures": self._migration_failures,
            "session_affinity": session_map,
            "session_states": session_state,
            "migration_pending": {
                sid: data
                for sid, data in session_state.items()
                if data.get("state") == _STATE_MIGRATION_PENDING
            },
            "kv_migration_enabled": _HTTPX_AVAILABLE and bool(self._full_url),
        }

    # DS-6: Dynamic quarter management for QuarterScheduler

    def add_quarter(self, backend: Any) -> int:
        """Add a quarter backend instance. Returns the new quarter index.

        Thread-safe. The new quarter starts receiving requests immediately.
        """
        with self._lock:
            idx = len(self._quarters)
            self._quarters.append(backend)
            self._quarter_active.append(False)
            self._quarter_urls.append(_get_base_url(backend))
        logger.info(
            "ConcurrencyAware[%s]: added quarter %d (now %d quarters)",
            self._role, idx, len(self._quarters),
        )
        return idx

    def remove_quarter(self, idx: int) -> bool:
        """Remove a quarter backend by index. Returns False if index invalid.

        Thread-safe. Refuses removal if the quarter has active requests.
        Caller must drain traffic before calling this.
        Also cleans up any session affinity pointing to this quarter.
        """
        with self._lock:
            if idx < 0 or idx >= len(self._quarters):
                return False
            if self._quarter_active[idx]:
                logger.warning(
                    "ConcurrencyAware[%s]: refusing to remove active quarter %d",
                    self._role, idx,
                )
                return False
            self._quarters.pop(idx)
            self._quarter_active.pop(idx)
            self._quarter_urls.pop(idx)
            # Fix up session affinity: remove stale references, shift indices
            stale = [sid for sid, qidx in self._session_quarter.items() if qidx == idx]
            for sid in stale:
                del self._session_quarter[sid]
            for sid in list(self._session_quarter):
                if self._session_quarter[sid] > idx:
                    self._session_quarter[sid] -= 1
        logger.info(
            "ConcurrencyAware[%s]: removed quarter %d (now %d quarters)",
            self._role, idx, len(self._quarters),
        )
        return True

    def quarter_count(self) -> int:
        """Return current number of quarter instances."""
        with self._lock:
            return len(self._quarters)

    def is_quarter_active(self, idx: int) -> bool:
        """Check if a specific quarter has active requests."""
        with self._lock:
            if 0 <= idx < len(self._quarter_active):
                return self._quarter_active[idx]
            return False

    def __len__(self) -> int:
        return 1 + len(self._quarters)

    def __repr__(self) -> str:
        return (
            f"ConcurrencyAwareBackend(role={self._role!r}, "
            f"full=1, quarters={len(self._quarters)}, "
            f"migrations={self._migrations})"
        )
