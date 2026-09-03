"""Sizing tests for scripts/utils/numa_evict.py — the FORCING form (INF-70/C7).

Run with:

    python3 -m pytest scripts/utils/test_numa_evict.py -v

MUTATION HARNESS. These tests must FAIL against the weak ``TARGET - free``
sizing that under-evicted all session on 2026-09-02 (D8x root cause). Prove
they have teeth by running them against that form once:

    NUMA_EVICT_TEST_FORM=weak python3 -m pytest scripts/utils/test_numa_evict.py

which swaps `plan_allocation_gib` for the weak formula via the autouse fixture
below. The expected result of that run is failures in every test decorated
`@teeth`; a green run under the weak form means the tests check nothing.
(Recorded 2026-09-03: forcing form 20 passed; weak form 14 failed / 6 passed.)

No subprocess runs here. The kernel is modelled by `_FakeBox`: allocating and
touching G GiB under membind on a node with F GiB free reclaims max(0, G - F)
of page cache, so after release the node holds max(F, G) GiB free. That is
exactly why the weak form fails — its G <= F never reclaims anything.
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numa_evict as ev  # noqa: E402

TARGET = 40
GIB_MB = 1024


def teeth(fn):
    """Marks a test that MUST fail under NUMA_EVICT_TEST_FORM=weak."""
    fn.teeth = True
    return fn


def _weak_plan(free_mb: int, target_gib: int, headroom_gib: int = 2) -> int:
    """The 2026-09-02 sizing: TARGET - free (+1). Freed nothing useful."""
    return max(0, target_gib - free_mb // GIB_MB + 1)


@pytest.fixture(autouse=True)
def _maybe_mutate_to_weak_form(monkeypatch):
    if os.environ.get("NUMA_EVICT_TEST_FORM") == "weak":
        monkeypatch.setattr(ev, "plan_allocation_gib", _weak_plan)
    yield


class _FakeBox:
    """Per-node free memory under the allocate-touch-release model."""

    def __init__(self, free_gib: dict[int, float]):
        self.free_mb = {n: int(g * GIB_MB) for n, g in free_gib.items()}
        self.calls: list[tuple[int, int]] = []
        self.after_pass_hook = None  # optional: simulate concurrent cache growth

    def query(self) -> dict[int, int]:
        return dict(self.free_mb)

    def evict(self, node: int, gib: int) -> bool:
        self.calls.append((node, gib))
        # G <= F: satisfied from free pages, nothing reclaimed, F unchanged.
        # G  > F: kernel reclaims G - F of page cache; after release free == G.
        self.free_mb[node] = max(self.free_mb[node], gib * GIB_MB)
        return True


# --- plan_allocation_gib: the sizing rule ------------------------------------

@teeth
def test_node_just_below_target_forces_target_plus_headroom():
    assert ev.plan_allocation_gib((TARGET - 1) * GIB_MB, TARGET) == TARGET + 2


@teeth
@pytest.mark.parametrize("free_mb", [TARGET * GIB_MB, TARGET * GIB_MB + 1, 200 * GIB_MB])
def test_node_at_or_above_target_allocates_nothing(free_mb):
    assert ev.plan_allocation_gib(free_mb, TARGET) == 0


@teeth
@pytest.mark.parametrize("free_gib", [0, 1, 10, 20, 30, 39])
def test_forced_allocation_always_exceeds_what_is_free(free_gib):
    """The whole point: the allocation must not fit in the free pages."""
    gib = ev.plan_allocation_gib(free_gib * GIB_MB, TARGET)
    assert gib > free_gib
    assert gib >= TARGET + 1


def test_headroom_is_configurable():
    assert ev.plan_allocation_gib(0, TARGET, headroom_gib=5) == TARGET + 5


# --- run_eviction: passes and verification -----------------------------------

@teeth
def test_near_target_nodes_reach_target_in_one_pass():
    box = _FakeBox({0: 39, 1: 20, 2: 45, 3: 39.9})
    short, allocs = ev.run_eviction(
        [0, 1, 2, 3], TARGET, query_free_mb=box.query, evict=box.evict, log=lambda *_: None
    )
    assert short == []
    assert allocs == [(1, 0, 42), (1, 1, 42), (1, 3, 42)]
    assert box.free_mb[2] == 45 * GIB_MB  # untouched
    assert all(box.free_mb[n] >= TARGET * GIB_MB for n in (0, 1, 3))


@teeth
def test_nodes_already_at_target_get_no_allocation():
    box = _FakeBox({0: 40, 1: 41, 2: 60, 3: 100})
    short, allocs = ev.run_eviction(
        [0, 1, 2, 3], TARGET, query_free_mb=box.query, evict=box.evict, log=lambda *_: None
    )
    assert short == []
    assert allocs == []
    assert box.calls == []


@teeth
def test_concurrent_cache_growth_after_pass_one_triggers_pass_two():
    """B4/D7a shape: a download refilled node 3 between eviction and load."""
    box = _FakeBox({0: 39, 1: 39, 2: 39, 3: 39})
    passes_seen = {"n": 0}
    real_query = box.query

    def query_with_growth():
        # Called once before pass 1, then once after each pass. Simulate a
        # writer stealing node 3 right after pass 1 completed.
        passes_seen["n"] += 1
        if passes_seen["n"] == 2:
            box.free_mb[3] = 35 * GIB_MB
        return real_query()

    short, allocs = ev.run_eviction(
        [0, 1, 2, 3], TARGET, query_free_mb=query_with_growth, evict=box.evict,
        log=lambda *_: None,
    )
    assert short == []
    assert [a for a in allocs if a[0] == 1] == [(1, 0, 42), (1, 1, 42), (1, 2, 42), (1, 3, 42)]
    assert [a for a in allocs if a[0] == 2] == [(2, 3, 42)]


def test_gives_up_after_configured_passes_and_reports_short():
    class _Stuck(_FakeBox):
        def evict(self, node, gib):
            self.calls.append((node, gib))
            return True  # the child ran, the memory never came back

    box = _Stuck({0: 39, 1: 50})
    short, allocs = ev.run_eviction(
        [0, 1], TARGET, query_free_mb=box.query, evict=box.evict, passes=3,
        log=lambda *_: None,
    )
    assert short == [0]
    assert allocs == [(1, 0, 42), (2, 0, 42), (3, 0, 42)]


def test_dry_run_plans_pass_one_and_allocates_nothing():
    box = _FakeBox({0: 10, 1: 50})
    short, allocs = ev.run_eviction(
        [0, 1], TARGET, query_free_mb=box.query, evict=box.evict, dry_run=True,
        log=lambda *_: None,
    )
    assert allocs == [(1, 0, 42)]
    assert box.calls == []
    assert box.free_mb[0] == 10 * GIB_MB


# --- main(): exit codes end to end (numactl and the child are faked) ---------

def _wire_main(monkeypatch, box: _FakeBox):
    def fake_hw():
        return "".join(f"node {n} free: {mb} MB\n" for n, mb in sorted(box.free_mb.items()))

    monkeypatch.setattr(ev, "numactl_hardware", fake_hw)
    monkeypatch.setattr(ev, "evict_node", lambda node, gib, timeout_s: box.evict(node, gib))


@teeth
def test_main_exit_0_when_forcing_reaches_target(monkeypatch, capsys):
    box = _FakeBox({0: 39, 1: 39, 2: 39, 3: 39})
    _wire_main(monkeypatch, box)
    assert ev.main(["--target-gib", str(TARGET)]) == 0
    assert box.calls == [(0, 42), (1, 42), (2, 42), (3, 42)]
    assert "OK: every requested node" in capsys.readouterr().out


def test_main_exit_1_when_still_short(monkeypatch, capsys):
    box = _FakeBox({0: 39, 1: 39})
    monkeypatch.setattr(box, "evict", lambda node, gib: True)  # child ran, nothing freed
    _wire_main(monkeypatch, box)
    assert ev.main(["--target-gib", str(TARGET)]) == 1
    assert "still below" in capsys.readouterr().err


def test_main_exit_2_on_bad_passes(monkeypatch):
    _wire_main(monkeypatch, _FakeBox({0: 39}))
    assert ev.main(["--passes", "0"]) == 2
    assert ev.main(["--target-gib", "199", "--headroom-gib", "2"]) == 2


def test_main_dry_run_exit_0_and_no_child(monkeypatch, capsys):
    box = _FakeBox({0: 1, 1: 1})
    _wire_main(monkeypatch, box)
    assert ev.main(["--dry-run"]) == 0
    assert box.calls == []
    assert "dry run" in capsys.readouterr().out


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
