"""Disk-leak audit 2026-09-27 item 3: KV-migration slot files are removed.

Offline only: every "server" is a monkeypatched slot helper that writes the
file into a tmp_path slot dir the way llama-server would under
``--slot-save-path``. Covers terminal success, each abort kind, the
no-reuse-on-retry contract, the sweep age boundary, symlink/outside-dir
refusal, and in-flight exclusion.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

import src.backends.concurrency_aware as CA
from src.backends import kv_slot_files as KS
from src.scheduling.migration_transaction import MigrationState


class _Stub:
    def __init__(self, url: str) -> None:
        self.config = type("C", (), {"base_url": url})()


SWEEP_OFF = KS.KvSlotRetentionPolicy(sweep_enabled=False)


def _backend(slot_dir: Path, policy: KS.KvSlotRetentionPolicy = SWEEP_OFF):
    return CA.ConcurrencyAwareBackend(
        _Stub("http://full.invalid:1"),
        [_Stub("http://q0.invalid:2"), _Stub("http://q1.invalid:3")],
        role="frontdoor",
        slot_save_dir=slot_dir,
        kv_slot_policy=policy,
    )


def _fake_server(monkeypatch, slot_dir: Path, *, save=5, restore=5, seen=None):
    """Patch the slot helpers: save writes the file, restore reads it."""

    def _save(url, slot_id=0, filename=None):
        if save not in (None, False):
            (slot_dir / filename).write_bytes(b"x" * 64)
        return save

    def _restore(url, slot_id=0, filename=None):
        if seen is not None:
            seen.append((filename, (slot_dir / filename).exists()))
        return restore

    monkeypatch.setattr(CA, "_slot_save", _save)
    monkeypatch.setattr(CA, "_slot_restore", _restore)
    monkeypatch.setattr(CA, "_slot_erase", lambda *a, **k: True)


def _files(slot_dir: Path) -> list[str]:
    return sorted(p.name for p in slot_dir.iterdir())


# ── forward migration: terminal outcomes ─────────────────────────────────


def test_committed_migration_unlinks_file_after_restore(tmp_path, monkeypatch):
    seen: list = []
    _fake_server(monkeypatch, tmp_path, seen=seen)
    be = _backend(tmp_path)
    txn = be._migrate_kv("sess-a", 0)
    assert txn.state is MigrationState.COMMITTED
    # The restore read the file (it existed at restore time) ...
    assert seen and seen[0][1] is True
    # ... and it is gone once the transaction is terminal.
    assert _files(tmp_path) == []
    assert be._inflight_slot_files == set()


def test_save_failed_abort_leaves_no_file(tmp_path, monkeypatch):
    # A 0-token save is an HTTP 200 that still writes a (tiny) file.
    def _save(url, slot_id=0, filename=None):
        (tmp_path / filename).write_bytes(b"x")
        return 0

    monkeypatch.setattr(CA, "_slot_save", _save)
    monkeypatch.setattr(CA, "_slot_restore", lambda *a, **k: pytest.fail("no restore"))
    be = _backend(tmp_path)
    txn = be._migrate_kv("sess-a", 0)
    assert txn.state is MigrationState.ABORTED and txn.detail == "save_failed"
    assert _files(tmp_path) == []


def test_token_mismatch_abort_unlinks_file(tmp_path, monkeypatch):
    _fake_server(monkeypatch, tmp_path, save=10, restore=3)
    be = _backend(tmp_path)
    txn = be._migrate_kv("sess-a", 0)
    assert txn.detail == "restore_token_mismatch"
    assert _files(tmp_path) == []


def test_restore_failed_abort_keeps_file_for_sweep(tmp_path, monkeypatch):
    """restore_failed may be a client timeout while the server still reads the file."""
    _fake_server(monkeypatch, tmp_path, restore=None)
    be = _backend(tmp_path)
    txn = be._migrate_kv("sess-a", 0)
    assert txn.detail == "restore_failed"
    kept = _files(tmp_path)
    assert len(kept) == 1 and kept[0].startswith("kv_migrate_frontdoor_sess-a_")
    # Terminal → no longer in flight, so the age-gated sweep may reclaim it later.
    assert be._inflight_slot_files == set()


def test_retry_never_reuses_a_prior_file(tmp_path, monkeypatch):
    """A retry is a NEW transaction with a new filename; the kept file is never re-read."""
    restored_names: list = []
    _fake_server(monkeypatch, tmp_path, restore=None, seen=restored_names)
    be = _backend(tmp_path)
    be._migrate_kv("sess-a", 0)
    first = _files(tmp_path)
    _fake_server(monkeypatch, tmp_path, seen=restored_names)
    txn2 = be._migrate_kv("sess-a", 0)
    assert txn2.state is MigrationState.COMMITTED
    assert restored_names[1][0] not in first
    # The retry removed only its own file; the earlier orphan waits for the sweep.
    assert _files(tmp_path) == first


def test_unlink_disabled_by_policy_keeps_files(tmp_path, monkeypatch):
    _fake_server(monkeypatch, tmp_path)
    be = _backend(tmp_path, KS.KvSlotRetentionPolicy(unlink_on_terminal=False))
    be._migrate_kv("sess-a", 0)
    assert len(_files(tmp_path)) == 1


def test_file_unlinked_even_when_commit_raises(tmp_path, monkeypatch):
    _fake_server(monkeypatch, tmp_path)
    be = _backend(tmp_path)

    def _boom(*a, **k):
        raise RuntimeError("bookkeeping failed")

    monkeypatch.setattr(be, "_finalize_quarter_assignment", _boom)
    with pytest.raises(RuntimeError):
        be._migrate_kv("sess-a", 0)
    # The restore had completed (VERIFIED) — nothing can still read the file.
    assert _files(tmp_path) == []
    assert be._inflight_slot_files == set()


def test_unlink_never_blocks_migration_when_dir_missing(tmp_path, monkeypatch):
    missing = tmp_path / "nope"
    monkeypatch.setattr(CA, "_slot_save", lambda *a, **k: 5)
    monkeypatch.setattr(CA, "_slot_restore", lambda *a, **k: 5)
    monkeypatch.setattr(CA, "_slot_erase", lambda *a, **k: True)
    be = _backend(missing)
    assert be._migrate_kv("sess-a", 0).state is MigrationState.COMMITTED


# ── reverse migration ────────────────────────────────────────────────────


def test_reverse_migration_committed_unlinks_file(tmp_path, monkeypatch):
    seen: list = []
    _fake_server(monkeypatch, tmp_path, seen=seen)
    be = _backend(tmp_path)
    be._session_quarter["sess-a"] = 1
    be._reverse_migrate_kv("sess-a", 1)
    assert seen and seen[0][1] is True
    assert _files(tmp_path) == []
    assert be._inflight_slot_files == set()


def test_reverse_migration_restore_failed_keeps_file(tmp_path, monkeypatch):
    _fake_server(monkeypatch, tmp_path, restore=None)
    be = _backend(tmp_path)
    be._reverse_migrate_kv("sess-a", 1)
    assert len(_files(tmp_path)) == 1


# ── unlink confinement ───────────────────────────────────────────────────


def test_unlink_refuses_non_migration_and_pathy_names(tmp_path):
    (tmp_path / "slot_0_abc.bin").write_bytes(b"prefix-cache")
    outside = tmp_path.parent / f"{tmp_path.name}-outside"
    outside.mkdir()
    (outside / "kv_migrate_x.bin").write_bytes(b"x")
    assert not KS.unlink_slot_file(tmp_path, "slot_0_abc.bin", reason="t")
    assert not KS.unlink_slot_file(tmp_path, f"../{outside.name}/kv_migrate_x.bin", reason="t")
    assert not KS.unlink_slot_file(tmp_path, "kv_migrate_x.txt", reason="t")
    assert (tmp_path / "slot_0_abc.bin").exists()
    assert (outside / "kv_migrate_x.bin").exists()


def test_unlink_refuses_symlink_and_never_touches_its_target(tmp_path):
    outside = tmp_path.parent / f"{tmp_path.name}-target"
    outside.mkdir()
    target = outside / "precious.bin"
    target.write_bytes(b"keep")
    link = tmp_path / "kv_migrate_link.bin"
    link.symlink_to(target)
    assert not KS.unlink_slot_file(tmp_path, link.name, reason="t")
    assert link.is_symlink() and target.read_bytes() == b"keep"


def test_unlink_through_symlinked_slot_dir_stays_in_real_dir(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    (real / "kv_migrate_a.bin").write_bytes(b"x")
    alias = tmp_path / "alias"
    alias.symlink_to(real)
    assert KS.unlink_slot_file(alias, "kv_migrate_a.bin", reason="t")
    assert not (real / "kv_migrate_a.bin").exists()


# ── sweep ────────────────────────────────────────────────────────────────


def _aged(path: Path, age_s: float, now: float) -> None:
    path.write_bytes(b"x" * 10)
    os.utime(path, (now - age_s, now - age_s))


def test_sweep_age_boundary(tmp_path):
    now = 1_900_000_000.0
    _aged(tmp_path / "kv_migrate_old.bin", 3600.0, now)       # exactly N → eligible
    _aged(tmp_path / "kv_migrate_young.bin", 3599.0, now)     # 1 s short → kept
    res = KS.sweep_orphan_slot_files(tmp_path, min_age_s=3600.0, now=now)
    assert [n for n, _ in res.removed] == ["kv_migrate_old.bin"]
    assert res.skipped_young == 1
    assert _files(tmp_path) == ["kv_migrate_young.bin"]


def test_sweep_only_touches_migration_bins_in_that_dir(tmp_path):
    now = 1_900_000_000.0
    _aged(tmp_path / "slot_0_abcdef.bin", 99999, now)          # prefix cache
    _aged(tmp_path / "kv_migrate_a.txt", 99999, now)
    (tmp_path / "manifest.json").write_text("{}")
    sub = tmp_path / "sub"
    sub.mkdir()
    _aged(sub / "kv_migrate_nested.bin", 99999, now)           # no recursion
    res = KS.sweep_orphan_slot_files(tmp_path, min_age_s=60, now=now)
    assert res.removed == []
    assert (sub / "kv_migrate_nested.bin").exists()
    assert (tmp_path / "slot_0_abcdef.bin").exists()


def test_sweep_refuses_symlink_to_outside_dir(tmp_path):
    now = time.time()
    slot = tmp_path / "slot"
    slot.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    target = outside / "kv_migrate_victim.bin"
    _aged(target, 99999, now)
    (slot / "kv_migrate_victim.bin").symlink_to(target)
    res = KS.sweep_orphan_slot_files(slot, min_age_s=60)
    assert res.removed == [] and res.refused == 1
    assert target.exists() and (slot / "kv_migrate_victim.bin").is_symlink()


def test_sweep_excludes_in_flight_names(tmp_path):
    now = 1_900_000_000.0
    _aged(tmp_path / "kv_migrate_live.bin", 99999, now)
    _aged(tmp_path / "kv_migrate_dead.bin", 99999, now)
    res = KS.sweep_orphan_slot_files(
        tmp_path, min_age_s=60, in_flight={"kv_migrate_live.bin"}, now=now,
    )
    assert [n for n, _ in res.removed] == ["kv_migrate_dead.bin"]
    assert res.skipped_in_flight == 1
    assert _files(tmp_path) == ["kv_migrate_live.bin"]


def test_sweep_dry_run_and_cap(tmp_path):
    now = 1_900_000_000.0
    for i in range(5):
        _aged(tmp_path / f"kv_migrate_{i}.bin", 99999, now)
    dry = KS.sweep_orphan_slot_files(tmp_path, min_age_s=60, dry_run=True, now=now)
    assert len(dry.stale) == 5 and dry.removed == [] and len(_files(tmp_path)) == 5
    capped = KS.sweep_orphan_slot_files(tmp_path, min_age_s=60, max_files=2, now=now)
    assert len(capped.removed) == 2 and capped.truncated
    assert len(_files(tmp_path)) == 3


def test_backend_sweep_skips_while_a_migration_is_in_flight(tmp_path):
    old = time.time() - 7200
    for name in ("kv_migrate_frontdoor_a_111.bin", "kv_migrate_frontdoor_b_222.bin"):
        p = tmp_path / name
        p.write_bytes(b"x")
        os.utime(p, (old, old))
    be = _backend(tmp_path, KS.KvSlotRetentionPolicy(sweep_enabled=True))
    # Construction swept both (both old, nothing in flight).
    assert _files(tmp_path) == []
    p = tmp_path / "kv_migrate_frontdoor_c_333.bin"
    p.write_bytes(b"x")
    os.utime(p, (old, old))
    be._inflight_slot_files.add("kv_migrate_frontdoor_d_444.bin")
    assert be._maybe_sweep_slot_files(force=True) is None  # any in-flight → no sweep
    assert _files(tmp_path) == ["kv_migrate_frontdoor_c_333.bin"]
    be._inflight_slot_files.clear()
    be._maybe_sweep_slot_files(force=True)
    assert _files(tmp_path) == []


def test_backend_sweep_disabled_is_report_only(tmp_path, caplog):
    old = time.time() - 7200
    p = tmp_path / "kv_migrate_frontdoor_a_111.bin"
    p.write_bytes(b"x")
    os.utime(p, (old, old))
    with caplog.at_level("WARNING"):
        _backend(tmp_path)  # sweep disabled (default)
    assert p.exists()
    assert any("kv_migrate" in r.getMessage() and "stale" in r.getMessage() for r in caplog.records)


def test_backend_sweep_rate_limited(tmp_path):
    be = _backend(tmp_path, KS.KvSlotRetentionPolicy(sweep_enabled=True))
    old = time.time() - 7200
    p = tmp_path / "kv_migrate_frontdoor_z_999.bin"
    p.write_bytes(b"x")
    os.utime(p, (old, old))
    be._maybe_sweep_slot_files()  # within interval of the construction sweep → no-op
    assert p.exists()


# ── policy ───────────────────────────────────────────────────────────────


def test_repo_policy_parses_with_operator_enabled_sweep():
    # The repo policy file carries the operator's 2026-09-27 decision (sweep ON);
    # the SAFE fallback for an unusable file stays OFF (test below).
    pol = KS.load_policy()
    assert pol.unlink_on_terminal is True
    assert pol.sweep_enabled is True
    assert pol.sweep_min_age_minutes == 60


def test_policy_rejects_unknown_keys_and_tiny_age():
    with pytest.raises(ValueError):
        KS.parse_policy({"version": 1, "bogus": 1})
    with pytest.raises(ValueError):
        KS.parse_policy({"version": 1, "sweep": {"min_age_minutes": 1}})


def test_unusable_policy_falls_back_to_safe_default(tmp_path):
    bad = tmp_path / "p.yaml"
    bad.write_text("version: 2\n")
    assert KS.load_policy(bad) == KS.SAFE_DEFAULT_POLICY
    assert KS.SAFE_DEFAULT_POLICY.sweep_enabled is False


# ── hermetic seam (tests/unit/conftest.py::_hermetic_kv_slot_io) ─────────


def test_unit_suite_never_reaches_live_slot_api(monkeypatch):
    """Regression: the old-sess leak — unit tests saved/erased slots on live :8070."""

    def _no_network(*a, **k):
        raise AssertionError("unit test reached the live llama-server slot API")

    if CA._HTTPX_AVAILABLE:
        monkeypatch.setattr(CA.httpx, "post", _no_network)
    be = CA.ConcurrencyAwareBackend(
        _Stub("http://localhost:8070"),
        [_Stub("http://localhost:8080"), _Stub("http://localhost:8180")],
        role="frontdoor",
        full_port=8070,
    )
    txn = be._migrate_kv("old-sess", 0)
    assert txn.state is MigrationState.ABORTED and txn.detail == "save_failed"
    # And the real kv_slots dir is never resolved, listed or swept.
    assert be._slot_save_dirs() == []
