"""serial_run accepts --new-anchor-epoch / --floor-* without touching any existing binding.

NOT YET RUN (no test window granted). Pure functions only; no process, no store.
    cd scripts/kernel_rnd && python3 -m pytest autokernel/loop/test_serial_anchor_epoch.py -q
"""
from __future__ import annotations

from autokernel.loop import serial_run as sr

BASE = ["--resolved-campaign", "/nonexistent-rc.json", "--worktree", "/w", "--store", "/s",
        "--planner-model", "m"]
EXTRA = ["--new-anchor-epoch", "a" * 40, "--floor-registry", "/r", "--floor-numerics-epoch", "n1",
         "--floor-require-grade", "decision", "--floor-co-tenancy", "quiet"]


def test_flags_are_declared_changing_so_bindings_ignore_them():
    for flag in (*sr.ANCHOR_EPOCH_FLAGS, *sr.FLOOR_REGISTRY_FLAGS):
        assert flag in sr._CHANGING_FLAGS
    assert sr._without(BASE + EXTRA, sr._CHANGING_FLAGS) == sr._without(BASE, sr._CHANGING_FLAGS)


def test_epoch_equals_form_is_stripped_too():
    assert sr._without(BASE + ["--new-anchor-epoch=" + "a" * 40], sr.ANCHOR_EPOCH_FLAGS) == BASE


def test_flags_absent_leaves_argv_untouched():
    assert sr._without(BASE, sr.ANCHOR_EPOCH_FLAGS) == BASE
