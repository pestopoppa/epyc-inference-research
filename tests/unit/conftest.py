"""Hermetic seam for `tests/unit/`: keep unit tests off the shared host's
real inference lock and contention-gate state (TD-21).

WHY THIS EXISTS
----------------
A unit test that builds a real `LLMPrimitives` with MOCKED backends still
runs the real, unmocked `_real_call()` control flow (`src/llm_primitives/
inference.py`). That path always reaches two pieces of genuinely
process/host-wide state before it ever touches a backend:

1. The cross-role contention gate (`src.scheduling.contention_gate.get_gate()`),
   whose `_active_holders()` reads `src.runtime.cpu_region_lock.
   active_region_holders()` -- a scan of the HOST's real `/proc/locks` /
   region-lock files -- whenever `ORCHESTRATOR_PER_REGION_LOCKS` is truthy.
2. The host-wide `fcntl` inference lock (`src.inference_lock.inference_lock`,
   real module `src.runtime.inference_lock`), which by default resolves to
   `<ORCHESTRATOR_PATHS_TMP_DIR or {llm_root}/tmp>/heavy_model.lock` --
   i.e. `/mnt/raid0/llm/tmp/heavy_model.lock` on this shared host -- and
   blocks (default 180s timeout) on whatever REAL, concurrently-running
   session currently holds it.

On a shared host with other sessions actively serving inference, that means
a unit test can block for 60s+ on a lock/gate state that has nothing to do
with the code under test (observed cross-session flakiness).

WHY THIS IS SURGICAL, NOT A BLANKET ENV OVERRIDE
--------------------------------------------------
An earlier version of this fixture forced `ORCHESTRATOR_PATHS_TMP_DIR` and
`ORCHESTRATOR_PER_REGION_LOCKS` via `monkeypatch.setenv` + `reset_config()`.
That is the wrong seam: `PathsConfig` (`src/config/models.py`) is ONE
dataclass shared by everything -- `models_dir`, `cache_dir`,
`registry_path`, etc. -- so overriding the env global rippled into every
OTHER `get_config()` consumer active during the same test, not just the
lock. It broke tests with no relationship to the lock at all:
`tmp_path`-listing tests (this fixture's own scratch directory became an
extra, unexpected entry in `tmp_path.iterdir()`),
`test_config_consolidation.py::test_all_paths_on_raid` (asserts the REAL
default paths), `test_autokernel_enrollment.py` (compares a fresh
`get_config()`-derived recompute against values cached at collection time,
which the env change desynced), and any test that itself replaces
`src.config.get_config` with a plain (uncached) callable for its own
`monkeypatch` (this fixture's `reset_config()` teardown then called
`.cache_clear()` on a plain function and raised `AttributeError`).

This version instead monkeypatches ONLY the `get_config` NAME BINDING
inside `src.runtime.inference_lock`'s own module namespace (which
`src.inference_lock` is the exact same module object as -- see the TD-21
concurrent-import fix and its identity contract) to a wrapper that
delegates to the real, current `get_config()` for everything except
`paths.tmp_dir`. `from X import Y` binds an independent name in the
importing module; patching this one binding cannot reach any OTHER
module's own `get_config` import, so nothing outside the lock's own path
resolution is affected -- no env var, no shared lru_cache invalidation.

Equivalently, the contention gate singleton (`get_gate()`) is replaced with
one constructed with `active_holders_fn=lambda: {}` -- exactly the
existing, already-supported test seam `tests/unit/contention_gate_fixture.
py` uses for its own dedicated gate tests -- rather than forcing
`ORCHESTRATOR_PER_REGION_LOCKS`, so no test that itself inspects that env
var for an unrelated reason is affected either.

OPTING OUT
----------
A test that intentionally exercises the real lock/gate wiring -- e.g. to
prove the DEFAULT (unpatched) `get_config()`/env resolution behaviour, or
the `ORCHESTRATOR_PER_REGION_LOCKS=1` host-read branch itself -- marks
itself ``@pytest.mark.real_inference_lock_state`` to skip this fixture's
overrides entirely and control its own environment. As of TD-21, no
EXISTING test needs this: `tests/unit/test_inference_lock.py` and
`tests/unit/test_scheduling_contention_gate.py` (+ friends using
`tests/unit/contention_gate_fixture.py`) already fully control their own
`get_config`/`active_holders_fn` (applied after this fixture, via the same
function-scoped `monkeypatch`, so their more specific override wins); the
marker exists so a FUTURE test of the real default path has a documented
way to ask for it.
"""

from __future__ import annotations

import dataclasses

import pytest


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "real_inference_lock_state: opt out of the tests/unit/ hermetic "
        "inference-lock/contention-gate fixture and exercise the real "
        "env-derived lock path / gate host reads.",
    )
    config.addinivalue_line(
        "markers",
        "real_kv_slot_io: opt out of the tests/unit/ hermetic KV-slot fixture "
        "(real llama-server slot HTTP helpers + real kv_slots directory).",
    )
    config.addinivalue_line(
        "markers",
        "real_inference_tap: opt out of the tests/unit/ hermetic inference-tap "
        "fixture (real INFERENCE_TAP_* env + real on-disk tap sentinel).",
    )


@pytest.fixture(scope="session")
def _hermetic_inference_tap_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("hermetic_inference_tap")


@pytest.fixture(autouse=True)
def _hermetic_inference_tap(request, monkeypatch, _hermetic_inference_tap_dir):
    """Keep unit tests out of the LIVE production inference tap.

    `src.runtime.inference_tap.is_active()` is true when EITHER the
    `INFERENCE_TAP_FILE` env var is set OR the on-disk sentinel
    (`<paths.tmp_dir>/.inference_tap_active`, written by the autopilot TUI and
    left in place on this host) names a tap file. A unit test that drives the
    real `LLMPrimitives._real_call()` with mocked backends therefore appended
    fake sections (`ROLE=role_a`/`role_b`, `prompt A`, `edit prompt`, ...) to
    `/mnt/raid0/llm/tmp/inference_tap.log` and its `inference_tap_events.jsonl`
    -- the files the dashboard reads as live traffic (738 sections from pytest
    PIDs by 2026-09-28). This fixture clears the env keys and points the
    module's cached sentinel path at a file that never exists, so the tap is
    OFF by default. A test that exercises the tap sets `INFERENCE_TAP_FILE`
    (or its own `_SENTINEL`) itself; its monkeypatch applies after this one.
    """
    if request.node.get_closest_marker("real_inference_tap") is not None:
        yield
        return
    import src.runtime.inference_tap as tap_mod

    monkeypatch.delenv("INFERENCE_TAP_FILE", raising=False)
    monkeypatch.delenv("INFERENCE_TAP_EVENTS_FILE", raising=False)
    monkeypatch.setattr(
        tap_mod, "_SENTINEL", str(_hermetic_inference_tap_dir / ".inference_tap_active")
    )
    monkeypatch.setattr(tap_mod, "_sentinel_cache", ("", 0.0))
    yield


@pytest.fixture(autouse=True)
def _hermetic_kv_slot_io(request, monkeypatch):
    """Keep unit tests off the LIVE llama-servers' slot API and slot directory.

    `ConcurrencyAwareBackend` tests build backends on `http://localhost:8070`
    (+ 8080/8180), which are the live frontdoor full and halves on this host.
    Any test that reached `_migrate_kv` without stubbing the slot helpers made
    the live :8070 server write a ~66 MB `kv_migrate_frontdoor_old-sess_*.bin`
    into `/mnt/raid0/llm/cache/kv_slots/frontdoor`, restored it on a live half
    and ERASED live slot 0 — two files per suite run, 142 files / 11.2 GiB by
    2026-09-27 (disk-leak audit item 3). The helpers are replaced with offline
    doubles that fail closed (a save that fails aborts the migration cold), and
    the slot-dir resolver returns nothing so no backend ever lists or sweeps
    the real directory. A test that stubs the helpers itself overrides these
    (its monkeypatch applies after this fixture).
    """
    if request.node.get_closest_marker("real_kv_slot_io") is not None:
        yield
        return
    import src.backends.concurrency_aware as ca_mod
    import src.backends.kv_slot_files as ks_mod

    monkeypatch.setattr(ca_mod, "_slot_save", lambda *a, **k: None)
    monkeypatch.setattr(ca_mod, "_slot_restore", lambda *a, **k: None)
    monkeypatch.setattr(ca_mod, "_slot_erase", lambda *a, **k: False)
    monkeypatch.setattr(ks_mod, "default_slot_save_dirs", lambda *roles: [])
    yield


@pytest.fixture(autouse=True)
def _hermetic_inference_lock_and_contention_gate(request, tmp_path_factory, monkeypatch):
    if request.node.get_closest_marker("real_inference_lock_state") is not None:
        yield
        return

    import src.runtime.inference_lock as rt_lock
    from src.scheduling import contention_gate as gate_mod

    # A dedicated tmp dir OUTSIDE this test's own `tmp_path`, so a test that
    # asserts an exact directory listing / file count on its own `tmp_path`
    # never sees this fixture's scratch directory as a surprise extra entry.
    lock_dir = tmp_path_factory.mktemp("hermetic_inference_lock")

    real_get_config = rt_lock.get_config

    def _patched_get_config():
        cfg = real_get_config()
        patched_paths = dataclasses.replace(cfg.paths, tmp_dir=lock_dir)
        return dataclasses.replace(cfg, paths=patched_paths)

    monkeypatch.setattr(rt_lock, "get_config", _patched_get_config)

    previous_gate = gate_mod._GATE_SINGLETON
    with gate_mod._GATE_SINGLETON_LOCK:
        gate_mod._GATE_SINGLETON = gate_mod.ContentionGate(active_holders_fn=lambda: {})

    try:
        yield lock_dir
    finally:
        with gate_mod._GATE_SINGLETON_LOCK:
            gate_mod._GATE_SINGLETON = previous_gate
