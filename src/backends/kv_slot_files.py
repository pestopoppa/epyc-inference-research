"""On-disk lifecycle of KV-migration slot files (``kv_migrate_*.bin``).

``ConcurrencyAwareBackend._migrate_kv`` asks llama-server to ``save`` a slot to
``<--slot-save-path>/<filename>`` and a sibling server to ``restore`` it. The
HTTP ``erase`` that follows clears only the SERVER'S IN-MEMORY slot; nothing
ever removed the file, so every migration left ~66 MB behind (disk-leak audit
2026-09-27, item 3: 142 files / 11.2 GiB under ``kv_slots/frontdoor``).

This module owns the two removal paths:

* ``unlink_slot_file`` — the migration's own file, once its transaction is
  terminal and no restore can still be reading it.
* ``sweep_orphan_slot_files`` — a bounded, age-gated sweep of files a
  migration could not remove itself (process died mid-migration, restore
  timed out client-side while the server was still reading).

Both are confined to ONE directory: only bare ``kv_migrate_*.bin`` names, only
regular files (symlinks are refused, never followed, never removed), resolved
through the directory's realpath and removed with ``dir_fd`` so a name can
never reach outside it. Prefix-cache slot files (``slot_<id>_<hash>.bin``,
``src/inference/prefix_cache.py``) share the directory and are never touched.

Every removal is best-effort: a failure is logged and never raised into the
migration path.

Policy is data: ``orchestration/kv_slot_retention_policy.yaml``.
"""

from __future__ import annotations

import logging
import os
import stat
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

logger = logging.getLogger(__name__)

POLICY_PATH = (
    Path(__file__).resolve().parents[2] / "orchestration" / "kv_slot_retention_policy.yaml"
)

MIGRATION_SLOT_PREFIX = "kv_migrate_"
MIGRATION_SLOT_SUFFIX = ".bin"


@dataclass(frozen=True)
class KvSlotRetentionPolicy:
    """Retention knobs. Defaults are the SAFE posture used when the file is unusable."""

    unlink_on_terminal: bool = True
    sweep_enabled: bool = False
    sweep_min_age_minutes: float = 60.0
    sweep_interval_minutes: float = 10.0
    sweep_max_files: int = 500

    @property
    def sweep_min_age_s(self) -> float:
        return self.sweep_min_age_minutes * 60.0

    @property
    def sweep_interval_s(self) -> float:
        return self.sweep_interval_minutes * 60.0


SAFE_DEFAULT_POLICY = KvSlotRetentionPolicy()

_SWEEP_KEYS = {"enabled", "min_age_minutes", "interval_minutes", "max_files_per_sweep"}


def parse_policy(document: Any) -> KvSlotRetentionPolicy:
    """Validate a parsed policy document. Raises ValueError on the first problem."""
    if not isinstance(document, dict):
        raise ValueError("kv_slot_retention_policy: document must be a mapping")
    unknown = sorted(set(document) - {"version", "unlink_on_terminal", "sweep"})
    if unknown:
        raise ValueError(f"kv_slot_retention_policy: unknown top-level key(s): {unknown}")
    if document.get("version") != 1:
        raise ValueError("kv_slot_retention_policy: version must be 1")
    unlink = document.get("unlink_on_terminal", True)
    if not isinstance(unlink, bool):
        raise ValueError("kv_slot_retention_policy: unlink_on_terminal must be a boolean")
    sweep = document.get("sweep") or {}
    if not isinstance(sweep, dict):
        raise ValueError("kv_slot_retention_policy: sweep must be a mapping")
    unknown = sorted(set(sweep) - _SWEEP_KEYS)
    if unknown:
        raise ValueError(f"kv_slot_retention_policy: unknown key(s) in sweep: {unknown}")
    enabled = sweep.get("enabled", False)
    if not isinstance(enabled, bool):
        raise ValueError("kv_slot_retention_policy: sweep.enabled must be a boolean")

    def _num(key: str, default: float, minimum: float) -> float:
        value = sweep.get(key, default)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value < minimum:
            raise ValueError(
                f"kv_slot_retention_policy: sweep.{key} must be a number >= {minimum}"
            )
        return float(value)

    # Floor of 5 minutes: a migration is bounded by the 30 s save + 30 s restore
    # client timeouts, so anything younger may still be in flight somewhere.
    min_age = _num("min_age_minutes", 60.0, 5.0)
    interval = _num("interval_minutes", 10.0, 0.0)
    max_files = sweep.get("max_files_per_sweep", 500)
    if isinstance(max_files, bool) or not isinstance(max_files, int) or max_files < 1:
        raise ValueError(
            "kv_slot_retention_policy: sweep.max_files_per_sweep must be an int >= 1"
        )
    return KvSlotRetentionPolicy(
        unlink_on_terminal=unlink,
        sweep_enabled=enabled,
        sweep_min_age_minutes=min_age,
        sweep_interval_minutes=interval,
        sweep_max_files=max_files,
    )


def load_policy(path: Path | None = None) -> KvSlotRetentionPolicy:
    """Load the policy file; an unreadable/invalid file falls back to the SAFE default
    (sweep disabled) with a warning — disk hygiene must never break serving."""
    policy_path = POLICY_PATH if path is None else Path(path)
    try:
        import yaml

        return parse_policy(yaml.safe_load(policy_path.read_text()))
    except Exception as exc:  # noqa: BLE001 - fail to the safe posture, loudly
        logger.warning(
            "kv slot retention policy unusable (%s: %s); using safe default "
            "(unlink_on_terminal=%s, sweep disabled)",
            policy_path, exc, SAFE_DEFAULT_POLICY.unlink_on_terminal,
        )
        return SAFE_DEFAULT_POLICY


def default_slot_save_dirs(*roles: str) -> list[Path]:
    """Candidate ``--slot-save-path`` directories for a backend's roles.

    Mirrors the launch side exactly: ``orchestrator_stack.py`` passes
    ``cache.slot_save_path``, which ``stack_priors`` compiles to
    ``SLOT_SAVE_DIR / primary_role`` (``SLOT_SAVE_DIR = cache_dir / "kv_slots"``,
    both from ``ORCHESTRATOR_PATHS_CACHE_DIR``), and falls back to the same
    ``SLOT_SAVE_DIR / role`` when unset.
    """
    try:
        from src.config import get_config

        base = Path(get_config().paths.cache_dir) / "kv_slots"
    except Exception as exc:  # noqa: BLE001
        logger.debug("kv slot dir resolution failed: %s", exc)
        return []
    out: list[Path] = []
    for role in roles:
        if role and "/" not in role and role not in {".", ".."}:
            cand = base / role
            if cand not in out:
                out.append(cand)
    return out


def is_migration_slot_name(name: str) -> bool:
    """True only for a bare ``kv_migrate_*.bin`` name (no separators, not hidden)."""
    return (
        isinstance(name, str)
        and name.startswith(MIGRATION_SLOT_PREFIX)
        and name.endswith(MIGRATION_SLOT_SUFFIX)
        and len(name) > len(MIGRATION_SLOT_PREFIX) + len(MIGRATION_SLOT_SUFFIX)
        and "/" not in name
        and "\\" not in name
        and "\x00" not in name
        and os.path.basename(name) == name
    )


def _open_dir(slot_dir: str | os.PathLike[str]) -> tuple[int, str] | None:
    root = os.path.realpath(os.fspath(slot_dir))
    if not os.path.isdir(root):
        return None
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    return os.open(root, flags), root


def _unlink_in_dir(dir_fd: int, root: str, name: str, *, reason: str) -> int | None:
    """Unlink ``name`` inside the already-open ``root``. Returns bytes freed, None if refused/absent."""
    try:
        st = os.stat(name, dir_fd=dir_fd, follow_symlinks=False)
    except FileNotFoundError:
        return None
    if stat.S_ISLNK(st.st_mode):
        logger.warning(
            "kv slot file %s/%s is a symlink; refusing to remove or follow it (%s)",
            root, name, reason,
        )
        return None
    if not stat.S_ISREG(st.st_mode):
        logger.warning("kv slot path %s/%s is not a regular file; refusing (%s)", root, name, reason)
        return None
    # Belt and braces: the name is bare and the directory fd is the realpath,
    # so the entry cannot resolve anywhere but ``root``.
    if os.path.dirname(os.path.realpath(os.path.join(root, name))) != root:
        logger.warning("kv slot file %s/%s resolves outside its directory; refusing", root, name)
        return None
    os.unlink(name, dir_fd=dir_fd)
    logger.info(
        "kv slot file removed: %s/%s (%d bytes, reason=%s)", root, name, st.st_size, reason,
    )
    return int(st.st_size)


def unlink_slot_file(
    slot_dir: str | os.PathLike[str], filename: str, *, reason: str
) -> bool:
    """Best-effort removal of one migration slot file from ``slot_dir``. Never raises."""
    if not is_migration_slot_name(filename):
        logger.warning(
            "kv slot unlink refused: %r is not a bare %s*%s name (%s)",
            filename, MIGRATION_SLOT_PREFIX, MIGRATION_SLOT_SUFFIX, reason,
        )
        return False
    try:
        opened = _open_dir(slot_dir)
        if opened is None:
            return False
        dir_fd, root = opened
        try:
            return _unlink_in_dir(dir_fd, root, filename, reason=reason) is not None
        finally:
            os.close(dir_fd)
    except Exception as exc:  # noqa: BLE001 - best-effort by contract
        logger.warning("kv slot unlink failed for %s/%s: %s", slot_dir, filename, exc)
        return False


@dataclass
class SweepResult:
    slot_dir: str
    dry_run: bool
    removed: list[tuple[str, int]] = field(default_factory=list)   # (name, bytes)
    stale: list[tuple[str, int, float]] = field(default_factory=list)  # (name, bytes, age_s)
    skipped_young: int = 0
    skipped_in_flight: int = 0
    refused: int = 0
    truncated: bool = False

    @property
    def removed_bytes(self) -> int:
        return sum(b for _, b in self.removed)

    @property
    def stale_bytes(self) -> int:
        return sum(b for _, b, _ in self.stale)


def sweep_orphan_slot_files(
    slot_dir: str | os.PathLike[str],
    *,
    min_age_s: float,
    in_flight: Iterable[str] = (),
    max_files: int = 500,
    dry_run: bool = False,
    now: float | None = None,
) -> SweepResult:
    """Remove ``kv_migrate_*.bin`` files in exactly ``slot_dir`` aged >= ``min_age_s``.

    Skips: names in ``in_flight``, anything younger than ``min_age_s``, every
    non-matching name, and every non-regular entry (symlinks are refused and
    never followed). Does not recurse. ``dry_run`` only reports what is stale.
    At most ``max_files`` files are removed per call. Never raises.
    """
    result = SweepResult(slot_dir=os.fspath(slot_dir), dry_run=dry_run)
    exclude = set(in_flight)
    clock = time.time() if now is None else now
    try:
        opened = _open_dir(slot_dir)
        if opened is None:
            return result
        dir_fd, root = opened
        result.slot_dir = root
        try:
            with os.scandir(dir_fd) as it:
                names = sorted(e.name for e in it)
            for name in names:
                if not is_migration_slot_name(name):
                    continue
                if name in exclude:
                    result.skipped_in_flight += 1
                    continue
                try:
                    st = os.stat(name, dir_fd=dir_fd, follow_symlinks=False)
                except FileNotFoundError:
                    continue
                if not stat.S_ISREG(st.st_mode):
                    result.refused += 1
                    logger.warning(
                        "kv slot sweep: %s/%s is not a regular file (symlink?); refusing",
                        root, name,
                    )
                    continue
                age = clock - st.st_mtime
                if age < min_age_s:
                    result.skipped_young += 1
                    continue
                result.stale.append((name, int(st.st_size), age))
                if dry_run:
                    continue
                if len(result.removed) >= max_files:
                    result.truncated = True
                    continue
                try:
                    freed = _unlink_in_dir(
                        dir_fd, root, name, reason=f"orphan_sweep age={age / 60:.0f}m"
                    )
                except OSError as exc:
                    logger.warning("kv slot sweep: unlink %s/%s failed: %s", root, name, exc)
                    freed = None
                if freed is not None:
                    result.removed.append((name, freed))
        finally:
            os.close(dir_fd)
    except Exception as exc:  # noqa: BLE001 - best-effort by contract
        logger.warning("kv slot sweep of %s failed: %s", slot_dir, exc)
    return result


__all__ = [
    "KvSlotRetentionPolicy",
    "MIGRATION_SLOT_PREFIX",
    "MIGRATION_SLOT_SUFFIX",
    "POLICY_PATH",
    "SAFE_DEFAULT_POLICY",
    "SweepResult",
    "default_slot_save_dirs",
    "is_migration_slot_name",
    "load_policy",
    "parse_policy",
    "sweep_orphan_slot_files",
    "unlink_slot_file",
]
