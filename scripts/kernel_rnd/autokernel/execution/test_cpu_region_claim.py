#!/usr/bin/env python3
"""Unit tests for cpu_region_claim.py — ACQUISITION of a CPU region claim.

NO INFERENCE, NO BENCHMARK, NO BUILD, NO MODEL. Nothing here pins a core, runs a
llama binary, or touches the live `/mnt/raid0/llm/tmp` namespace: every test uses
a per-test temp lock root, and the "cores" are arithmetic over integers.

WHY THESE TESTS SPAWN REAL PROCESSES
------------------------------------
The claim's entire content is "another PROCESS cannot take these cores", and the
mechanism is a kernel object (`flock` on an open file description) plus `/proc`
liveness. A mocked lock or a mocked `/proc` would test the test. So exclusivity,
crash recovery and the interoperation with the orchestrator's raw `flock` are all
asserted against real `subprocess` children that really take real locks, and the
crash test really SIGKILLs one.

Every child process is one this test created; the test never touches, signals, or
name-pattern-matches any pre-existing process on this shared host — it signals a
PID it captured from its own `Popen`. Every child carries a hard self-timeout so
a failed assertion cannot leave a lock holder behind, and `tearDown` terminates
and reaps whatever is left.

Run standalone:
    python3 -W error::ResourceWarning -m unittest \\
        scripts/kernel_rnd/autokernel/execution/test_cpu_region_claim.py
"""
from __future__ import annotations

import ast
import fcntl
import importlib.util
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

# Import through the PACKAGE, never by putting this directory on sys.path — the
# sibling `resource` package would shadow the stdlib `resource` module for
# anything imported afterwards (AutoPilot scar item 12: ambient import identity).
_KERNEL_RND = str(Path(__file__).resolve().parents[2])
if _KERNEL_RND not in sys.path:
    sys.path.insert(0, _KERNEL_RND)

from autokernel.execution import cpu_region_claim as crc  # noqa: E402
from autokernel.resource import preflight as pf  # noqa: E402
from autokernel import schemas as S  # noqa: E402

CAMPAIGN = "ak-test-20260803"
CHILD_MAX_LIFE_S = 45.0

#: The orchestrator's topology module, checked when it is present on this host.
ORCHESTRATOR_TOPOLOGY = Path(
    "/mnt/raid0/llm/epyc-orchestrator/src/runtime/instance_topology.py")

_CHILD_SOURCE = '''\
"""Child worker for test_cpu_region_claim.py. Takes REAL locks in a REAL process."""
import fcntl, json, os, sys, time

with open(sys.argv[1]) as _fh:
    cfg = json.load(_fh)
sys.path.insert(0, cfg["kernel_rnd"])
from autokernel.execution import cpu_region_claim as crc

deadline = time.time() + cfg["max_life_s"]


def _emit(name, obj):
    tmp = cfg["workdir"] + "/" + name + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(obj, fh)
    os.replace(tmp, cfg["workdir"] + "/" + name)


def _stopped():
    return os.path.exists(cfg["workdir"] + "/stop")


mode = cfg["mode"]

if mode == "raw_flock":
    # EXACTLY what epyc-orchestrator/src/runtime/cpu_region_lock.py does: open
    # 'a+b', flock LOCK_EX, write the schema_version-1 attribution payload.
    # Nothing of this module is used, so a passing exclusion test proves the two
    # implementations meet on the kernel object and not on shared code.
    handles = []
    for path in cfg["raw_paths"]:
        fh = open(path, "a+b")
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        payload = {
            "schema_version": 1, "pid": os.getpid(), "role": cfg["raw_role"],
            "region": path.rsplit(".", 2)[1], "regions": cfg["raw_regions"],
            "instance_idx": 0, "request_tag": "orchestrator-child",
            "started_at": time.time(),
        }
        fh.seek(0); fh.truncate(0)
        fh.write(json.dumps(payload, sort_keys=True).encode("utf-8")); fh.write(b"\\n")
        fh.flush()
        handles.append(fh)
    _emit("ready", {"pid": os.getpid(), "paths": cfg["raw_paths"]})
    while time.time() < deadline and not _stopped():
        time.sleep(0.02)
    for fh in handles:
        fh.seek(0); fh.truncate(0); fh.flush()
        fh.close()
    _emit("done", {"pid": os.getpid()})
    sys.exit(0)

journal = crc.RegionClaimJournal(cfg["journal"])

if mode == "contend":
    # Barrier: both racers wait for the parent's `go` file, so the contention is
    # real rather than an artefact of process start order.
    if cfg.get("barrier"):
        while not os.path.exists(cfg["barrier"]) and time.time() < deadline:
            time.sleep(0.005)
    try:
        claim = crc.acquire_cpu_region_claim(
            cfg["cpu_list"], purpose="contender", campaign_id=cfg["campaign"],
            journal=journal, role=cfg["role"], co_roles=cfg["co_roles"],
            timeout_s=cfg["timeout_s"], poll_s=0.02,
            stale_grace_s=cfg["stale_grace_s"], lock_root=cfg["lock_root"],
        )
    except crc.CpuRegionClaimError as exc:
        _emit("result", {"ok": False, "error_type": type(exc).__name__,
                         "error": str(exc),
                         "conflicts": getattr(exc, "conflicts", None)})
        sys.exit(0)
    held_from = time.time()
    time.sleep(cfg.get("hold_s") or 0.0)
    try:
        receipt = claim.receipt().to_dict()
    finally:
        claim.release()
    _emit("result", {"ok": True, "receipt": receipt,
                     "held_from": held_from, "held_to": time.time()})
    sys.exit(0)

claim = crc.acquire_cpu_region_claim(
    cfg["cpu_list"], purpose=cfg["purpose"], campaign_id=cfg["campaign"],
    journal=journal, role=cfg["role"], co_roles=cfg["co_roles"],
    timeout_s=cfg["timeout_s"], poll_s=0.02, stale_grace_s=cfg["stale_grace_s"],
    lock_root=cfg["lock_root"], holder_label=mode,
)
_emit("ready", {"pid": os.getpid(), "receipt": claim.receipt().to_dict()})

if mode == "hold_forever":
    # Waits to be SIGKILLed by the parent test. Bounded anyway.
    while time.time() < deadline:
        time.sleep(0.02)
    sys.exit(3)

try:
    while time.time() < deadline and not _stopped():
        time.sleep(0.02)
finally:
    receipt = claim.release()
_emit("done", {"receipt": receipt.to_dict()})
'''


class _ClaimTestBase(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="cpu_region_claim_test_"))
        self.lock_root = self.tmp / "locks"
        self.lock_root.mkdir()
        self.journal_path = self.tmp / "claims.jsonl"
        self.journal = crc.RegionClaimJournal(self.journal_path)
        self.child_script = self.tmp / "child_worker.py"
        self.child_script.write_text(_CHILD_SOURCE)
        self._children = []

    def tearDown(self):
        for _proc, workdir in self._children:
            log = self._child_log(workdir)
            self.assertNotIn("ResourceWarning", log,
                             f"a child leaked a handle in the claim path:\n{log}")
            self.assertNotIn("Exception ignored", log,
                             f"a child raised during cleanup:\n{log}")
        for proc, _workdir in self._children:
            if proc.poll() is None:
                # Only ever a process this test itself created, by captured PID.
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=5)
            else:
                proc.wait()
            self.assertIsNotNone(proc.poll(), "child process was not reaped")
        shutil.rmtree(self.tmp, ignore_errors=True)

    # -- child helpers ---------------------------------------------------
    def _spawn(self, mode, *, cpu_list="0-47", role="autokernel", co_roles=(),
               timeout_s=10.0, stale_grace_s=0.0, purpose="unit-test-hold",
               raw_paths=(), raw_role="frontdoor", raw_regions=(), name=None,
               barrier=None, hold_s=0.0):
        workdir = self.tmp / (name or f"child{len(self._children)}")
        workdir.mkdir()
        config = {
            "kernel_rnd": _KERNEL_RND,
            "workdir": str(workdir),
            "journal": str(self.journal_path),
            "lock_root": str(self.lock_root),
            "cpu_list": cpu_list,
            "role": role,
            "co_roles": list(co_roles),
            "mode": mode,
            "purpose": purpose,
            "campaign": CAMPAIGN,
            "timeout_s": timeout_s,
            "stale_grace_s": stale_grace_s,
            "max_life_s": CHILD_MAX_LIFE_S,
            "raw_paths": [str(p) for p in raw_paths],
            "raw_role": raw_role,
            "raw_regions": list(raw_regions),
            "barrier": str(barrier) if barrier is not None else None,
            "hold_s": hold_s,
        }
        (workdir / "config.json").write_text(json.dumps(config))
        log_path = workdir / "child.log"
        # The parent's copy of the log handle is closed immediately; the child
        # keeps its own dup. Leaving it open would trip -W error::ResourceWarning.
        with open(log_path, "wb") as log:
            proc = subprocess.Popen(
                [sys.executable, "-W", "error::ResourceWarning",
                 str(self.child_script), str(workdir / "config.json")],
                stdout=log, stderr=log, stdin=subprocess.DEVNULL,
            )
        self._children.append((proc, workdir))
        return proc, workdir

    def _child_log(self, workdir: Path) -> str:
        log = workdir / "child.log"
        if not log.exists():
            return "(no child log)"
        with open(log, "r", encoding="utf-8", errors="replace") as fh:
            return fh.read()

    def _await_file(self, path: Path, timeout_s=20.0, proc=None):
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if path.exists():
                with open(path, "r", encoding="utf-8") as fh:
                    return json.load(fh)
            if proc is not None and proc.poll() is not None and not path.exists():
                self.fail(f"child exited (rc={proc.returncode}) before writing {path.name}:\n"
                          f"{self._child_log(path.parent)}")
            time.sleep(0.02)
        self.fail(f"timed out waiting for {path}:\n{self._child_log(path.parent)}")

    def _stop_child(self, workdir: Path):
        (workdir / "stop").write_text("1")

    # -- assertions ------------------------------------------------------
    def _kinds(self):
        return [r["kind"] for r in self.journal.read_all()]

    def _records(self, kind):
        return [r for r in self.journal.read_all() if r["kind"] == kind]

    def _acquire(self, cpu_list="0-47", **kwargs):
        params = {
            "purpose": "unit-test",
            "campaign_id": CAMPAIGN,
            "journal": self.journal,
            "lock_root": self.lock_root,
            "timeout_s": 5.0,
            "poll_s": 0.02,
            "stale_grace_s": 0.0,
        }
        params.update(kwargs)
        return crc.acquire_cpu_region_claim(cpu_list, **params)

    def _lock_is_free(self, path: Path) -> bool:
        with open(path, "a+b") as fh:
            try:
                fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                return False
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
            return True

    def _write_payload(self, role: str, region: str, payload) -> Path:
        path = crc.region_lock_path(role, region, self.lock_root)
        path.write_text(json.dumps(payload, sort_keys=True) + "\n")
        return path

    def _autokernel_payload(self, *, holder, acquired_at=None, claim_id="akc-planted",
                            role="autokernel", region="q0"):
        return {
            "schema_version": 1, "pid": holder["pid"], "role": role, "region": region,
            "regions": [region], "instance_idx": None,
            "request_tag": f"autokernel:{claim_id}",
            "started_at": time.time(),
            "autokernel_schema": crc.CPU_REGION_CLAIM_SCHEMA,
            "claim_id": claim_id, "claim_role": role, "state": "held",
            "holder": holder, "cpu_list": "0-23", "physical_core_list": "0-23",
            "lock_roles": [role], "purpose": "planted", "campaign_id": CAMPAIGN,
            "acquired_at": acquired_at or crc._utc_now_iso(),
            "expires_at": None, "reclaimed_from": None,
        }


# =============================================================================
# Overlap arithmetic — the part that makes a claim system more than decorative
# =============================================================================

class TestOverlapArithmetic(unittest.TestCase):
    def test_overlap_is_not_equality(self):
        """0-95 and 48-143 are not equal, are not disjoint, and must conflict."""
        self.assertEqual(sorted(crc.cpu_lists_overlap("0-95", "48-143")),
                         ["q0", "q1", "q2", "q3"])
        # 48-143: 48-95 are physical, 96-143 fold onto 0-47. Both halves count.
        self.assertEqual(crc.cpu_list_to_regions("96-143"), ("q0", "q1"))

    def test_disjoint_halves_do_not_conflict(self):
        """The compliant path: the namespace is per-region so halves run together."""
        self.assertEqual(crc.cpu_lists_overlap("0-47", "48-95"), frozenset())
        self.assertEqual(crc.cpu_lists_overlap("0-23", "24-47"), frozenset())

    def test_a_single_shared_core_is_an_overlap(self):
        """One core in common is a conflict; region granularity must not round it away."""
        self.assertEqual(sorted(crc.cpu_lists_overlap("0-24", "24-47")), ["q1"])

    def test_smt_siblings_fold_onto_their_physical_core(self):
        """184-191 IS 88-95 (a different thread of the same eight cores).

        `feedback_mi210_host_threads_smt_siblings` records that GPU host threads
        belong on 184-191 and NOT on 88-95. That is a choice of which thread to
        pin, not a claim that the cores are free — so a GPU host claim and the
        canonical 0-95 CPU baseline CONFLICT, and the arithmetic has to say so.
        """
        self.assertEqual(crc.cpu_list_to_regions("184-191"), ("q3",))
        self.assertEqual(sorted(crc.cpu_lists_overlap("184-191", "0-95")), ["q3"])
        self.assertEqual(sorted(crc.cpu_lists_overlap("184-191", "72-95")), ["q3"])
        self.assertEqual(crc.cpu_lists_overlap("184-191", "0-71"), frozenset())

    def test_the_host_topology_confirms_the_sibling_fold(self):
        """The fold is a fact read from sysfs, not a constant typed from memory."""
        if not crc.SYSFS_CPU_ROOT.exists():
            self.skipTest("no /sys/devices/system/cpu on this host")
        mapping = crc.read_sibling_map()
        self.assertEqual(mapping.get(184), 88,
                         "cpu184's thread_siblings_list should anchor on physical core 88")
        self.assertEqual(mapping.get(88), 88)

    def test_an_unmappable_logical_cpu_is_refused_not_assumed(self):
        """Refusing beats guessing: a wrong fold under-covers the measurement."""
        with self.assertRaises(crc.CpuTopologyUnavailable):
            crc.physical_cores([200], sibling_map={})
        # Compliant path: the same id WITH a map resolves.
        self.assertEqual(crc.physical_cores([200], sibling_map={200: 12}), frozenset({12}))

    def test_a_sibling_anchored_outside_the_region_table_is_refused(self):
        with self.assertRaises(crc.CpuTopologyUnavailable):
            crc.physical_cores([200], sibling_map={200: 500})

    def test_cores_below_the_physical_max_need_no_sysfs(self):
        """0-95 must resolve on any host, so the canonical baseline is portable."""
        self.assertEqual(crc.physical_cores(range(0, 96), sibling_map={}),
                         frozenset(range(0, 96)))

    def test_cpu_list_parser_rejects_the_shapes_the_recipe_parser_rejects(self):
        for bad in ("", "  ", "0-", "-3", "5-1", "a-b", "0,,3", "0-4096", "x"):
            with self.assertRaises(ValueError, msg=f"{bad!r} should be refused"):
                crc.parse_cpu_list(bad)

    def test_cpu_list_parser_agrees_with_the_codified_recipe_parser(self):
        """Two parsers over one grammar is how one of them drifts.

        `evaluator.recipes._cpu_list_members` parses the `taskset -c` list of the
        constructed argv; this module parses the footprint a claim covers. If they
        disagreed, "the claim covers the measurement" would be uncheckable.
        """
        from autokernel.evaluator import recipes as R
        for spec in ("0-95", "184-191", "0-3,8-11", "7", "0-23,48-71"):
            self.assertEqual(crc.parse_cpu_list(spec),
                             frozenset(R._cpu_list_members(spec, field="t")),
                             f"parsers disagree on {spec!r}")

    def test_render_round_trips_a_cpu_list(self):
        for spec in ("0-95", "184-191", "0-3,8-11", "7"):
            self.assertEqual(crc.render_cpu_list(crc.parse_cpu_list(spec)), spec)

    def test_the_region_table_mirrors_the_orchestrator(self):
        """The mirror's drift trap. A change there fails a test here.

        `preflight.py` records why the orchestrator's topology is mirrored rather
        than imported (a second repo's import graph and sys.path ordering across
        three repos). The cost of mirroring is drift; this is the containment.
        """
        if not ORCHESTRATOR_TOPOLOGY.is_file():
            self.skipTest(f"{ORCHESTRATOR_TOPOLOGY} not present on this host")
        spec = importlib.util.spec_from_file_location(
            "_ak_orchestrator_topology_probe", ORCHESTRATOR_TOPOLOGY)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.assertEqual(dict(crc.REGION_CORE_RANGE),
                         {k: tuple(v) for k, v in module.REGION_CORE_RANGE.items()},
                         "the mirrored region table has drifted from the orchestrator's")
        self.assertEqual(tuple(crc.ATOMIC_REGIONS), tuple(module.ATOMIC_REGIONS))

    def test_region_mapping_matches_the_orchestrator_for_physical_cpu_lists(self):
        """For 0-95 the two implementations must agree exactly, list for list."""
        if not ORCHESTRATOR_TOPOLOGY.is_file():
            self.skipTest(f"{ORCHESTRATOR_TOPOLOGY} not present on this host")
        spec = importlib.util.spec_from_file_location(
            "_ak_orchestrator_topology_probe2", ORCHESTRATOR_TOPOLOGY)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for cpu_list in ("0-95", "0-47", "48-95", "0-23", "24-47,72-95"):
            self.assertEqual(set(crc.cpu_list_to_regions(cpu_list)),
                             set(module.cpu_list_to_regions(cpu_list)),
                             f"region mapping disagrees for {cpu_list!r}")

    def test_this_module_over_excludes_where_the_orchestrator_drops_siblings(self):
        """The documented divergence, asserted in the SAFE direction.

        `instance_topology.parse_cpu_list` drops logical cpus 96-191, so the
        orchestrator would take NO lock for 184-191. This module takes q3. The
        divergence must be a superset (over-exclusion costs concurrency;
        under-exclusion costs the measurement), never the other way.
        """
        if not ORCHESTRATOR_TOPOLOGY.is_file():
            self.skipTest(f"{ORCHESTRATOR_TOPOLOGY} not present on this host")
        spec = importlib.util.spec_from_file_location(
            "_ak_orchestrator_topology_probe3", ORCHESTRATOR_TOPOLOGY)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for cpu_list in ("184-191", "0-95", "96-143", "0-23,184-191"):
            theirs = set(module.cpu_list_to_regions(cpu_list))
            ours = set(crc.cpu_list_to_regions(cpu_list))
            self.assertTrue(theirs.issubset(ours),
                            f"{cpu_list!r}: ours {ours} must be a superset of theirs {theirs}")
        self.assertEqual(set(module.cpu_list_to_regions("184-191")), set())
        self.assertEqual(set(crc.cpu_list_to_regions("184-191")), {"q3"})


# =============================================================================
# The lock plan — the total order, provable without taking a lock
# =============================================================================

class TestLockPlan(_ClaimTestBase):
    def test_order_is_global_first_then_role_sorted_then_region_sorted(self):
        plan = crc.plan_region_claim("0-95", role="autokernel",
                                     co_roles=("frontdoor", "architect_general"),
                                     lock_root=self.lock_root)
        names = [(r, g) for r, g, _p in plan.lock_steps]
        self.assertEqual(names[:4], [("GLOBAL", q) for q in ("q0", "q1", "q2", "q3")])
        roles_in_order = []
        for role, _region in names[4:]:
            if not roles_in_order or roles_in_order[-1] != role:
                roles_in_order.append(role)
        self.assertEqual(roles_in_order, sorted({"autokernel", "frontdoor",
                                                 "architect_general"}))
        for role in roles_in_order:
            regions = [g for r, g in names if r == role]
            self.assertEqual(regions, sorted(regions), f"{role}'s regions are unsorted")

    def test_every_global_step_precedes_every_role_step(self):
        """The property that makes deadlock against the orchestrator impossible.

        `cpu_region_lock.cpu_region_lock` takes GLOBAL-all (region-sorted) then
        its own role's locks (region-sorted). Our order must be a linear
        extension of the same partial order, or two repos can hold each other's
        next lock.
        """
        plan = crc.plan_region_claim("0-95", role="ak", co_roles=("zzz", "aaa"),
                                     lock_root=self.lock_root)
        kinds = [r for r, _g, _p in plan.lock_steps]
        last_global = max(i for i, r in enumerate(kinds) if r == crc.GLOBAL_MUTEX_ROLE)
        first_role = min(i for i, r in enumerate(kinds) if r != crc.GLOBAL_MUTEX_ROLE)
        self.assertLess(last_global, first_role)

    def test_the_plan_is_what_acquisition_actually_walks(self):
        """A plan nothing consults would be documentation, not a guarantee."""
        plan = crc.plan_region_claim("0-47", role="autokernel", lock_root=self.lock_root)
        with self._acquire("0-47") as claim:
            self.assertEqual(list(claim.lock_paths),
                             [str(p) for _r, _g, p in plan.lock_steps])

    def test_roles_are_validated_never_rewritten(self):
        """The orchestrator maps '/'→'_'; that silently merges two distinct roles."""
        for bad in ("a/b", "a\\b", ".hidden", "", "x" * 80, 7):
            with self.assertRaises(ValueError, msg=f"{bad!r} should be refused"):
                crc.plan_region_claim("0-23", role=bad, lock_root=self.lock_root)
        # Compliant path: the shapes actually in the live namespace are accepted.
        for good in ("autokernel", "bench-canonical", "worker_general", "codex-m2-build"):
            crc.plan_region_claim("0-23", role=good, lock_root=self.lock_root)

    def test_the_global_pseudo_role_cannot_be_claimed_as_a_role(self):
        with self.assertRaises(ValueError):
            crc.plan_region_claim("0-23", role="GLOBAL", lock_root=self.lock_root)
        with self.assertRaises(ValueError):
            crc.plan_region_claim("0-23", role="ak", co_roles=("GLOBAL",),
                                  lock_root=self.lock_root)

    def test_region_lock_path_matches_the_orchestrator_naming_contract(self):
        path = crc.region_lock_path("frontdoor", "q2", self.lock_root)
        self.assertEqual(path.name, "cpu_region.frontdoor.q2.lock")
        self.assertEqual(crc.global_region_lock_path("q2", self.lock_root).name,
                         "cpu_region.GLOBAL.q2.lock")
        # preflight parses this shape; if it stops matching, claims become invisible.
        self.assertTrue(path.match(pf._LOCK_GLOB))

    def test_the_lock_root_is_resolved_by_the_orchestrators_precedence(self):
        env = {"ORCHESTRATOR_TMP_DIR": "/tmp/ak-root-a",
               "ORCHESTRATOR_PATHS_TMP_DIR": "/tmp/ak-root-b"}
        self.assertEqual(crc.default_region_lock_dir(env), Path("/tmp/ak-root-a"))
        self.assertEqual(crc.default_region_lock_dir({"ORCHESTRATOR_PATHS_TMP_DIR": "/tmp/b"}),
                         Path("/tmp/b"))
        self.assertEqual(crc.default_region_lock_dir({}), Path("/mnt/raid0/llm/tmp"))
        self.assertEqual(crc.default_region_lock_dir({}),
                         pf.default_region_lock_dir({}),
                         "the claim and the witness must resolve the SAME namespace")


# =============================================================================
# Acquire / hold / release
# =============================================================================

class TestAcquireHoldRelease(_ClaimTestBase):
    def test_a_claim_holds_every_lock_it_names_and_release_frees_them(self):
        claim = self._acquire("0-47")
        try:
            self.assertEqual(claim.regions, ("q0", "q1"))
            for path in claim.lock_paths:
                self.assertFalse(self._lock_is_free(Path(path)), f"{path} is not held")
            self.assertEqual(crc.check_region_claim_held(
                claim.receipt(), lock_root=self.lock_root).outcome, S.PASS)
        finally:
            claim.release()
        for path in claim.lock_paths:
            self.assertTrue(self._lock_is_free(Path(path)), f"{path} stayed locked")
        self.assertEqual(crc.check_region_claim_held(
            claim.receipt(), lock_root=self.lock_root).outcome, S.FAIL)

    def test_release_is_idempotent_and_journals_once(self):
        claim = self._acquire("0-23")
        claim.release()
        claim.release()
        self.assertEqual(self._kinds().count("claim_released"), 1)

    def test_the_context_manager_releases_on_an_exception(self):
        with self.assertRaises(RuntimeError):
            with crc.cpu_region_claim("0-23", purpose="boom", campaign_id=CAMPAIGN,
                                      journal=self.journal, lock_root=self.lock_root,
                                      timeout_s=5.0):
                raise RuntimeError("boom")
        self.assertTrue(self._lock_is_free(
            crc.global_region_lock_path("q0", self.lock_root)))

    def test_the_receipt_round_trips_and_refuses_a_mutated_one(self):
        with self._acquire("0-23") as claim:
            receipt = claim.receipt()
            self.assertEqual(crc.RegionClaimReceipt.from_dict(receipt.to_dict()), receipt)
            with self.assertRaises(ValueError):
                crc.RegionClaimReceipt.from_dict({**receipt.to_dict(), "extra": 1})
            broken = receipt.to_dict()
            del broken["regions"]
            with self.assertRaises(ValueError):
                crc.RegionClaimReceipt.from_dict(broken)

    def test_a_claim_requires_a_journal_and_an_attribution(self):
        with self.assertRaises(TypeError):
            self._acquire("0-23", journal=None)
        with self.assertRaises(ValueError):
            self._acquire("0-23", purpose="  ")
        with self.assertRaises(ValueError):
            self._acquire("0-23", campaign_id="")

    def test_the_lock_descriptors_are_not_inheritable(self):
        """An inherited descriptor makes a claim outlive its claimant, invisibly.

        The benchmark this claim exists for is a child process. If it inherited
        the lock fd, `release()` would return while the region stayed pinned by
        the child, and nothing would be able to reclaim it.

        HOW THE BITE WAS VERIFIED, and its limit: deleting `os.O_CLOEXEC` from
        `_open_lock_fd` does NOT fail this test, because CPython has made every
        descriptor non-inheritable by default since PEP 446 — an fdinfo-flag
        assertion here passed on the mutant and was therefore a fake guard, so it
        was deleted rather than kept. What this asserts is the property itself,
        which fails on the regression that IS reachable: a caller or a future
        edit calling `os.set_inheritable(fd, True)`.
        """
        with self._acquire("0-23") as claim:
            wanted = {os.path.realpath(p) for p in claim.lock_paths}
            checked = 0
            for entry in os.listdir("/proc/self/fd"):
                try:
                    target = os.readlink(f"/proc/self/fd/{entry}")
                except OSError:
                    continue
                if target not in wanted:
                    continue
                self.assertFalse(os.get_inheritable(int(entry)),
                                 f"the lock descriptor for {target} is inheritable")
                checked += 1
            self.assertEqual(checked, len(wanted),
                             "not every lock descriptor was found and checked")

    def test_a_child_spawned_under_a_claim_does_not_keep_it_alive(self):
        """The behavioural half: release must really free the region.

        A child is spawned WHILE the claim is held; the parent then releases and
        the region must be free even though the child is still running.
        """
        claim = self._acquire("0-23")
        proc, workdir = self._spawn("hold", cpu_list="48-71", role="unrelated")
        self._await_file(workdir / "ready", proc=proc)
        claim.release()
        self.assertIsNone(proc.poll(), "the child exited before the assertion")
        for path in claim.lock_paths:
            self.assertTrue(self._lock_is_free(Path(path)),
                            f"{path} stayed locked after release while a child was alive")
        self._stop_child(workdir)
        self._await_file(workdir / "done", proc=proc)

    def test_max_hold_is_advisory_and_expiry_is_reported_not_enforced(self):
        with self._acquire("0-23", max_hold_s=3600) as claim:
            receipt = claim.receipt()
            self.assertIsNotNone(receipt.expires_at)
            self.assertEqual(crc.check_claim_expiry(receipt).outcome, S.PASS)
            later = time.time() + 7200
            expired = crc.check_claim_expiry(receipt, now=later)
            self.assertEqual(expired.outcome, S.FAIL)
            self.assertIn("not reclaimable", " ".join(expired.reasons))
            # An expired claim is STILL held: nothing preempts a holder.
            self.assertEqual(crc.check_region_claim_held(
                receipt, lock_root=self.lock_root).outcome, S.PASS)

    def test_a_claim_without_max_hold_cannot_be_judged_expired(self):
        with self._acquire("0-23") as claim:
            self.assertEqual(crc.check_claim_expiry(claim.receipt()).outcome,
                             S.COULD_NOT_CHECK)

    def test_the_same_process_cannot_double_book_overlapping_regions(self):
        """flock conflicts between open file descriptions, so self-conflict is a
        clean timeout rather than a silently double-booked machine."""
        with self._acquire("0-23"):
            with self.assertRaises(crc.CpuRegionClaimTimeout):
                self._acquire("0-47", timeout_s=0)

    def test_the_same_process_may_hold_disjoint_regions(self):
        with self._acquire("0-23") as a:
            with self._acquire("48-71", role="autokernel-b") as b:
                self.assertEqual(a.regions, ("q0",))
                self.assertEqual(b.regions, ("q2",))


# =============================================================================
# Interoperation with the orchestrator's namespace
# =============================================================================

class TestOrchestratorInterop(_ClaimTestBase):
    def test_the_payload_is_read_by_the_orchestrator_contract(self):
        with self._acquire("0-23") as claim:
            path = crc.region_lock_path("autokernel", "q0", self.lock_root)
            payload = json.loads(path.read_text())
            self.assertEqual(payload["schema_version"],
                             crc.ORCHESTRATOR_PAYLOAD_SCHEMA_VERSION)
            self.assertEqual(payload["pid"], os.getpid())
            self.assertEqual(payload["role"], "autokernel")
            self.assertEqual(payload["region"], "q0")
            self.assertEqual(payload["regions"], ["q0"])
            self.assertIsInstance(payload["started_at"], float)
            self.assertIn(claim.claim_id, payload["request_tag"])

    def test_preflight_reads_an_autokernel_claim_as_held_with_no_notes(self):
        """The interop bite: `read_region_claims` is the fleet's reader.

        An unknown `schema_version`, a pid that disagrees with the flock holder,
        or a non-object payload each make it annotate the claim — which lands in
        a permanent evidence record. A clean read is the assertion.
        """
        with self._acquire("0-23") as claim:
            claims = pf.read_region_claims(self.lock_root, require_nonempty_namespace=False)
            mine = [c for c in claims if c.role == "autokernel" and c.region == "q0"]
            self.assertEqual(len(mine), 1)
            self.assertTrue(mine[0].held)
            self.assertFalse(mine[0].payload_is_stale)
            self.assertEqual(mine[0].notes, (), f"preflight annotated the claim: {mine[0].notes}")
            self.assertEqual(mine[0].payload["claim_id"], claim.claim_id)

    def test_the_preflight_note_check_actually_bites(self):
        """Control for the test above: a wrong schema_version DOES produce a note."""
        self._write_payload("autokernel", "q0", {"schema_version": 99, "pid": os.getpid()})
        claims = pf.read_region_claims(self.lock_root, require_nonempty_namespace=False)
        mine = [c for c in claims if c.role == "autokernel" and c.region == "q0"][0]
        self.assertTrue(any("schema_version" in n for n in mine.notes),
                        "preflight should have flagged an unknown payload version")

    def test_global_locks_are_held_without_a_payload(self):
        """The orchestrator's sweeper SKIPS GLOBAL files.

        A payload we left there after a crash is debris nothing in the fleet ever
        clears, so the GLOBAL layer is exclusion-only — exactly as
        `cpu_region_lock` writes it.
        """
        with self._acquire("0-47"):
            for region in ("q0", "q1"):
                path = crc.global_region_lock_path(region, self.lock_root)
                self.assertFalse(self._lock_is_free(path))
                self.assertEqual(path.read_text().strip(), "")

    def test_an_orchestrator_style_raw_flock_excludes_an_autokernel_claim(self):
        """Two implementations meeting on the kernel object, not on shared code.

        The child uses NONE of this module: it opens 'a+b' and calls
        `fcntl.flock(LOCK_EX)` exactly as `cpu_region_lock._acquire` does.
        """
        target = crc.global_region_lock_path("q0", self.lock_root)
        proc, workdir = self._spawn("raw_flock", raw_paths=[target],
                                    raw_regions=["q0"])
        self._await_file(workdir / "ready", proc=proc)
        with self.assertRaises(crc.CpuRegionClaimTimeout) as ctx:
            self._acquire("0-23", timeout_s=0.4)
        self.assertTrue(any(c.get("region") == "q0" for c in ctx.exception.conflicts))
        self._stop_child(workdir)
        self._await_file(workdir / "done", proc=proc)
        # Compliant path: once the raw holder is gone, the same claim succeeds.
        self._acquire("0-23", timeout_s=5.0).release()

    def test_co_roles_exclude_an_orchestrator_role_dispatch(self):
        """`co_roles` is what blocks a named role regardless of the GLOBAL flag.

        The orchestrator consults the GLOBAL layer only when started with
        `ORCHESTRATOR_CROSS_ROLE_DISJOINT_PLACEMENT`; holding its per-role lock
        blocks its dispatch unconditionally.
        """
        target = crc.region_lock_path("frontdoor", "q0", self.lock_root)
        proc, workdir = self._spawn("raw_flock", raw_paths=[target], raw_role="frontdoor",
                                    raw_regions=["q0"])
        self._await_file(workdir / "ready", proc=proc)
        # Without co_roles the frontdoor lock is not ours to take, so we succeed.
        plain = self._acquire("0-23", timeout_s=1.0)
        plain.release()
        # With co_roles the same claim must refuse while frontdoor is dispatching.
        with self.assertRaises(crc.CpuRegionClaimTimeout):
            self._acquire("0-23", co_roles=("frontdoor",), timeout_s=0.4)
        self._stop_child(workdir)
        self._await_file(workdir / "done", proc=proc)

    def test_dispatch_exclusion_names_the_roles_it_does_not_cover(self):
        crc.region_lock_path("frontdoor", "q0", self.lock_root).write_text("")
        with self._acquire("0-23") as claim:
            partial = crc.check_dispatch_exclusion(claim.receipt(), lock_root=self.lock_root)
            self.assertEqual(partial.outcome, S.COULD_NOT_CHECK)
            self.assertIn("frontdoor", " ".join(partial.reasons))
            self.assertIn(crc.ORCHESTRATOR_CROSS_ROLE_FLAG, " ".join(partial.reasons))
        with self._acquire("0-23", co_roles=("frontdoor",)) as claim:
            full = crc.check_dispatch_exclusion(claim.receipt(), lock_root=self.lock_root)
            self.assertEqual(full.outcome, S.PASS, full.reasons)

    def test_dispatch_exclusion_is_undecidable_when_the_claim_is_not_held(self):
        claim = self._acquire("0-23")
        receipt = claim.receipt()
        claim.release()
        self.assertEqual(
            crc.check_dispatch_exclusion(receipt, lock_root=self.lock_root).outcome, S.FAIL)


# =============================================================================
# Cross-process exclusion — the whole point, tested with real processes
# =============================================================================

class TestCrossProcessExclusion(_ClaimTestBase):
    def test_two_processes_cannot_hold_overlapping_regions(self):
        proc, workdir = self._spawn("hold", cpu_list="0-47")
        ready = self._await_file(workdir / "ready", proc=proc)
        with self.assertRaises(crc.CpuRegionClaimTimeout) as ctx:
            self._acquire("24-71", timeout_s=0.5)
        conflicts = ctx.exception.conflicts
        self.assertTrue(conflicts, "the timeout named no holder")
        self.assertTrue(any(ready["pid"] in c.get("holder_pids", []) for c in conflicts),
                        f"the child pid {ready['pid']} is not in {conflicts}")
        self._stop_child(workdir)
        self._await_file(workdir / "done", proc=proc)

    def test_two_processes_hold_disjoint_regions_at_the_same_time(self):
        """The compliant-path control: exclusion must not become a global mutex."""
        proc, workdir = self._spawn("hold", cpu_list="0-47")
        self._await_file(workdir / "ready", proc=proc)
        claim = self._acquire("48-95", timeout_s=5.0)
        try:
            self.assertIsNone(proc.poll(), "the child released before the assertion")
            self.assertEqual(claim.regions, ("q2", "q3"))
        finally:
            claim.release()
            self._stop_child(workdir)
        self._await_file(workdir / "done", proc=proc)

    def test_a_partially_overlapping_claim_is_refused_not_merely_an_equal_one(self):
        """0-95 held; 48-143 is neither equal nor disjoint and must lose."""
        proc, workdir = self._spawn("hold", cpu_list="0-95")
        self._await_file(workdir / "ready", proc=proc)
        with self.assertRaises(crc.CpuRegionClaimTimeout):
            self._acquire("48-143", timeout_s=0.4)
        self._stop_child(workdir)
        self._await_file(workdir / "done", proc=proc)

    def test_two_racing_processes_never_hold_the_region_at_the_same_time(self):
        """The guarantee stated the only way it can be checked: in TIME.

        Both children wait on a barrier file, then ask for the same region with a
        real budget, hold it for 0.3s, and report the interval over which they
        held it. Whichever order they get in, the two intervals must be disjoint;
        an overlap is a double-booked machine. Asserting "one wins" instead would
        be wrong — with a budget both are entitled to succeed, one after the
        other, and that is the correct behaviour.
        """
        barrier = self.tmp / "go"
        procs = [self._spawn("contend", cpu_list="0-23", timeout_s=20.0, hold_s=0.3,
                             role=f"racer{i}", name=f"racer{i}", barrier=barrier)
                 for i in range(2)]
        time.sleep(0.2)
        barrier.write_text("go")
        results = [self._await_file(w / "result", proc=p, timeout_s=40.0)
                   for p, w in procs]
        for result in results:
            self.assertTrue(result["ok"], result.get("error"))
        (a_from, a_to), (b_from, b_to) = [(r["held_from"], r["held_to"]) for r in results]
        self.assertTrue(a_to <= b_from or b_to <= a_from,
                        f"two processes held region q0 at the same time: "
                        f"[{a_from}, {a_to}] and [{b_from}, {b_to}]")

    def test_a_contender_with_no_budget_loses_to_a_held_claim(self):
        """`timeout_s=0` is ONE attempt: the loser fails as contention, not as a
        defect, and the holder is untouched."""
        with self._acquire("0-23") as claim:
            proc, workdir = self._spawn("contend", cpu_list="0-23", timeout_s=0.0,
                                        role="racer")
            result = self._await_file(workdir / "result", proc=proc)
            self.assertFalse(result["ok"])
            self.assertEqual(result["error_type"], "CpuRegionClaimTimeout", result["error"])
            self.assertEqual(crc.check_region_claim_held(
                claim.receipt(), lock_root=self.lock_root).outcome, S.PASS)

    def test_a_failed_acquisition_strands_no_lock(self):
        """A timeout mid-plan must release what it already took.

        The child holds q1 only; our plan takes GLOBAL q0 first and then blocks on
        GLOBAL q1. If the partial acquisition were not unwound, q0 would stay
        locked by a process that holds no claim — invisible, and unreclaimable
        while this process lives.
        """
        proc, workdir = self._spawn("hold", cpu_list="24-47")
        self._await_file(workdir / "ready", proc=proc)
        with self.assertRaises(crc.CpuRegionClaimTimeout):
            self._acquire("0-47", timeout_s=0.4)
        self.assertTrue(self._lock_is_free(crc.global_region_lock_path("q0", self.lock_root)),
                        "GLOBAL.q0 was stranded by a failed acquisition")
        self.assertTrue(self._lock_is_free(
            crc.region_lock_path("autokernel", "q0", self.lock_root)))
        self._stop_child(workdir)
        self._await_file(workdir / "done", proc=proc)

    def test_a_failed_acquisition_leaves_no_payload_behind(self):
        proc, workdir = self._spawn("hold", cpu_list="24-47")
        self._await_file(workdir / "ready", proc=proc)
        with self.assertRaises(crc.CpuRegionClaimTimeout):
            self._acquire("0-47", timeout_s=0.4)
        path = crc.region_lock_path("autokernel", "q0", self.lock_root)
        if path.exists():
            self.assertEqual(path.read_text().strip(), "",
                             "a failed acquisition left an attribution payload")
        self._stop_child(workdir)
        self._await_file(workdir / "done", proc=proc)


# =============================================================================
# Reclamation — and the three states, not two
# =============================================================================

class TestReclamation(_ClaimTestBase):
    def test_a_killed_holders_claim_is_reclaimed_and_the_reclaim_is_journaled(self):
        proc, workdir = self._spawn("hold_forever", cpu_list="0-23")
        ready = self._await_file(workdir / "ready", proc=proc)
        dead_pid = ready["pid"]
        self.assertEqual(dead_pid, proc.pid)
        # A PID this test captured from its own Popen. Never a name pattern.
        proc.send_signal(signal.SIGKILL)
        proc.wait(timeout=10)
        self.assertIsNotNone(proc.poll(), "the child was not reaped")
        # The kernel released the flock at exit; the payload is debris.
        path = crc.region_lock_path("autokernel", "q0", self.lock_root)
        self.assertNotEqual(path.read_text().strip(), "")
        claim = self._acquire("0-23", stale_grace_s=0.0)
        try:
            reclaims = self._records("claim_reclaimed")
            self.assertTrue(reclaims, "a reclamation happened with no journal record")
            record = reclaims[-1]["detail"]
            self.assertEqual(record["liveness"], "dead")
            self.assertEqual(record["reclaimed_from"]["holder"]["pid"], dead_pid)
            self.assertEqual(record["reclaimed_by_pid"], os.getpid())
            self.assertIsNotNone(claim.receipt().reclaimed_from)
        finally:
            claim.release()

    def test_a_dead_holder_inside_the_grace_period_is_waited_out_not_taken(self):
        proc, workdir = self._spawn("hold_forever", cpu_list="0-23")
        self._await_file(workdir / "ready", proc=proc)
        proc.send_signal(signal.SIGKILL)
        proc.wait(timeout=10)
        before = crc.region_lock_path("autokernel", "q0", self.lock_root).read_text()
        with self.assertRaises(crc.CpuRegionClaimTimeout):
            self._acquire("0-23", stale_grace_s=600.0, timeout_s=0.3)
        self.assertEqual(crc.region_lock_path("autokernel", "q0",
                                              self.lock_root).read_text(), before,
                         "a claim inside its grace period was modified")
        self.assertEqual(self._records("claim_reclaimed"), [])

    def test_a_live_holder_is_never_reclaimed(self):
        """A payload naming a LIVE process under a free lock is refused, not taken."""
        holder = crc.current_holder_identity("planted-live")
        self._write_payload("autokernel", "q0", self._autokernel_payload(holder=holder))
        with self.assertRaises(crc.CpuRegionClaimInconsistent) as ctx:
            self._acquire("0-23", stale_grace_s=0.0, timeout_s=0)
        self.assertIn("alive", str(ctx.exception))
        defects = self._records("defect")
        self.assertTrue(defects)
        self.assertEqual(defects[-1]["detail"]["defect_class"],
                         crc.DEFECT_LIVE_HOLDER_FREE_LOCK)

    def test_unknown_liveness_is_not_death(self):
        """Another host, a recycled-pid mismatch, a malformed holder: all UNKNOWN."""
        holder = crc.current_holder_identity()
        mutations = ({"host": "some-other-host"},
                     {"boot_id": None},
                     {"start_ticks": "not-an-int"},
                     {"pid": "not-an-int"})
        for index, mutation in enumerate(mutations):
            # A distinct role per case, so each plants into its OWN lock file and
            # the cases cannot mask one another.
            role = f"planted{index}"
            with self.subTest(mutation=mutation):
                planted = dict(holder)
                planted.update(mutation)
                self._write_payload(role, "q0",
                                    self._autokernel_payload(holder=planted, role=role))
                with self.assertRaises(crc.CpuRegionClaimInconsistent,
                                       msg=f"{mutation} should refuse, not reclaim"):
                    self._acquire("0-23", role=role, stale_grace_s=0.0, timeout_s=0)

    def test_a_recycled_pid_cannot_inherit_a_claim(self):
        """Same pid, different start_ticks ⇒ a DIFFERENT process ⇒ dead ⇒ reclaimable."""
        holder = dict(crc.current_holder_identity())
        holder["start_ticks"] = holder["start_ticks"] + 999999
        old = crc._utc_now_iso(time.time() - 3600)
        self._write_payload("autokernel", "q0",
                            self._autokernel_payload(holder=holder, acquired_at=old))
        claim = self._acquire("0-23", stale_grace_s=1.0)
        try:
            self.assertIn("recycled", self._records("claim_reclaimed")[-1]
                          ["detail"]["liveness_reason"])
        finally:
            claim.release()

    def test_an_undatable_claim_is_refused_rather_than_aged(self):
        holder = dict(crc.current_holder_identity())
        holder["start_ticks"] = holder["start_ticks"] + 999999   # dead
        payload = self._autokernel_payload(holder=holder)
        payload.pop("acquired_at")
        payload.pop("started_at")
        self._write_payload("autokernel", "q0", payload)
        with self.assertRaises(crc.CpuRegionClaimInconsistent) as ctx:
            self._acquire("0-23", stale_grace_s=0.0, timeout_s=0)
        self.assertIn("cannot be aged", str(ctx.exception))

    def test_orchestrator_debris_is_reclaimed_under_its_own_namespace_rule(self):
        """`cpu_region_lock` writes its payload only while holding the flock.

        Payload + free flock therefore means the writer was killed before its
        cleanup — which is exactly the state
        `sweep_stale_region_lock_payloads` clears without consulting pids.
        """
        self._write_payload("autokernel", "q0", {
            "schema_version": 1, "pid": 999999, "role": "autokernel", "region": "q0",
            "regions": ["q0"], "instance_idx": 0, "request_tag": "orchestrator",
            "started_at": time.time() - 3600,
        })
        claim = self._acquire("0-23", stale_grace_s=1.0)
        try:
            record = self._records("claim_reclaimed")[-1]["detail"]
            self.assertEqual(record["payload_kind"], "orchestrator")
            self.assertEqual(record["reclaimed_from"]["pid"], 999999)
        finally:
            claim.release()

    def test_orchestrator_debris_inside_the_grace_is_not_taken(self):
        self._write_payload("autokernel", "q0", {
            "schema_version": 1, "pid": 999999, "role": "autokernel", "region": "q0",
            "regions": ["q0"], "instance_idx": 0, "request_tag": "orchestrator",
            "started_at": time.time(),
        })
        with self.assertRaises(crc.CpuRegionClaimTimeout):
            self._acquire("0-23", stale_grace_s=600.0, timeout_s=0.2)

    def test_orchestrator_debris_that_cannot_be_dated_is_refused(self):
        self._write_payload("autokernel", "q0", {
            "schema_version": 1, "pid": 999999, "role": "autokernel", "region": "q0",
            "regions": ["q0"], "instance_idx": 0, "request_tag": "orchestrator",
        })
        with self.assertRaises(crc.CpuRegionClaimInconsistent) as ctx:
            self._acquire("0-23", stale_grace_s=0.0, timeout_s=0)
        self.assertIn("region-lock sweep", str(ctx.exception))

    def test_an_unparseable_payload_is_refused_not_reclaimed(self):
        path = crc.region_lock_path("autokernel", "q0", self.lock_root)
        path.write_text("{not json at all")
        with self.assertRaises(crc.CpuRegionClaimInconsistent):
            self._acquire("0-23", stale_grace_s=0.0, timeout_s=0)
        self.assertEqual(self._records("defect")[-1]["detail"]["defect_class"],
                         crc.DEFECT_UNVERIFIABLE_CLAIM)

    def test_a_payload_of_an_unknown_shape_is_refused(self):
        self._write_payload("autokernel", "q0", {"who": "knows", "schema_version": 42})
        with self.assertRaises(crc.CpuRegionClaimInconsistent) as ctx:
            self._acquire("0-23", stale_grace_s=0.0, timeout_s=0)
        self.assertIn("neither", str(ctx.exception))

    def test_an_empty_lock_file_is_simply_free(self):
        """The compliant-path control for every refusal above."""
        crc.region_lock_path("autokernel", "q0", self.lock_root).write_text("")
        self._acquire("0-23", stale_grace_s=0.0, timeout_s=0).release()


# =============================================================================
# Checkers — binding the claim to the measurement
# =============================================================================

class TestCheckers(_ClaimTestBase):
    def test_footprint_coverage_fails_when_the_claim_is_smaller_than_the_command(self):
        """P-AK-SEARCH-1 precondition 1: the claim must cover what argv pins."""
        with self._acquire("0-47") as claim:
            covered = crc.check_footprint_covered(claim.receipt(), "0-23")
            self.assertEqual(covered.outcome, S.PASS)
            uncovered = crc.check_footprint_covered(claim.receipt(), "0-95")
            self.assertEqual(uncovered.outcome, S.FAIL)
            self.assertIn("q2", " ".join(uncovered.reasons))

    def test_footprint_coverage_accepts_a_recipe_claim_footprint_object(self):
        class _Footprint:
            cpu_list = "0-23"
        with self._acquire("0-47") as claim:
            self.assertEqual(
                crc.check_footprint_covered(claim.receipt(), _Footprint()).outcome, S.PASS)

    def test_footprint_coverage_catches_the_smt_case(self):
        """A claim on 0-71 does NOT cover a command pinned to the GPU siblings."""
        with self._acquire("0-71") as claim:
            self.assertEqual(
                crc.check_footprint_covered(claim.receipt(), "184-191").outcome, S.FAIL)
        with self._acquire("72-95", role="ak-gpu") as claim:
            self.assertEqual(
                crc.check_footprint_covered(claim.receipt(), "184-191").outcome, S.PASS)

    def test_held_check_fails_a_receipt_whose_lock_is_free(self):
        claim = self._acquire("0-23")
        receipt = claim.receipt()
        claim.release()
        result = crc.check_region_claim_held(receipt, lock_root=self.lock_root)
        self.assertEqual(result.outcome, S.FAIL)
        self.assertIn("leaked", " ".join(result.reasons))

    def test_held_check_fails_when_the_payload_names_another_claim(self):
        with self._acquire("0-23") as claim:
            forged = {**claim.receipt().to_dict(), "claim_id": "akc-somebody-else"}
            self.assertEqual(
                crc.check_region_claim_held(forged, lock_root=self.lock_root).outcome, S.FAIL)

    def test_held_check_cannot_be_passed_by_a_receipt_naming_nothing(self):
        with self._acquire("0-23") as claim:
            empty = {**claim.receipt().to_dict(), "lock_paths": []}
            self.assertEqual(
                crc.check_region_claim_held(empty, lock_root=self.lock_root).outcome, S.FAIL)

    def test_the_canonical_baseline_footprint_comes_from_the_ratified_prefix(self):
        """Never retyped: `feedback_use_codified_recipes_not_memory`."""
        from autokernel.evaluator import recipes as R
        prefix = list(R.CANONICAL_PREFIX)
        self.assertEqual(crc.canonical_cpu_baseline_cpu_list(),
                         prefix[prefix.index("-c") + 1])
        self.assertEqual(crc.cpu_list_to_regions(crc.canonical_cpu_baseline_cpu_list()),
                         ("q0", "q1", "q2", "q3"),
                         "the canonical CPU baseline occupies the WHOLE machine")

    def test_the_gpu_host_footprint_is_the_sibling_range_and_it_conflicts(self):
        """184-191, not 88-95 — and it still contends for q3."""
        self.assertEqual(crc.gpu_host_cpu_list(), "184-191")
        self.assertEqual(crc.cpu_list_to_regions(crc.gpu_host_cpu_list()), ("q3",))
        self.assertEqual(
            sorted(crc.cpu_lists_overlap(crc.gpu_host_cpu_list(),
                                         crc.canonical_cpu_baseline_cpu_list())),
            ["q3"])

    def test_inspect_is_advisory_and_reports_the_holder(self):
        with self._acquire("0-23") as claim:
            view = crc.inspect_region_claims(self.lock_root)
            self.assertTrue(view["advisory"])
            self.assertIn("q0", view["held_regions"])
            entries = [e for e in view["regions"]["q0"] if e["role"] == "autokernel"]
            self.assertEqual(entries[0]["attribution"]["claim_id"], claim.claim_id)
            self.assertEqual(entries[0]["holder_liveness"], "live")

    def test_roles_present_lists_the_namespace(self):
        with self._acquire("0-23", co_roles=("frontdoor",)):
            self.assertEqual(set(crc.roles_present(self.lock_root)),
                             {"GLOBAL", "autokernel", "frontdoor"})


# =============================================================================
# Module hygiene — denial 8, asserted structurally
# =============================================================================

_FORBIDDEN_CALLS = {"kill", "killpg", "system", "popen", "spawnv", "spawnl", "pkill",
                    "pgrep", "killall", "send_signal", "terminate", "run", "Popen"}
_FORBIDDEN_IMPORTS = {"signal", "subprocess", "multiprocessing"}


def _docstring_nodes(tree: ast.AST) -> set:
    """ids of every docstring Constant node.

    Exempted from the string scan on purpose: `cpu_region_claim.py`'s docstring
    says its `/proc` read "is the opposite of `pgrep`", and a scanner that
    flagged the sentence explaining why a rule exists would push authors to stop
    explaining. The scan still bites on a string in CODE, which is where a shell
    command would live — `test_the_audit_bites` proves it.
    """
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                             ast.AsyncFunctionDef)):
            body = getattr(node, "body", None)
            if (body and isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)):
                out.add(id(body[0].value))
    return out


def _audit_no_process_control(source: str, filename: str) -> list:
    """Flag any process-control or name-pattern call in a module's own AST."""
    findings = []
    tree = ast.parse(source, filename=filename)
    docstrings = _docstring_nodes(tree)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] in _FORBIDDEN_IMPORTS:
                    findings.append(f"import {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            if (node.module or "").split(".")[0] in _FORBIDDEN_IMPORTS:
                findings.append(f"from {node.module} import ...")
        elif isinstance(node, ast.Call):
            func = node.func
            name = getattr(func, "attr", None) or getattr(func, "id", None)
            if name in _FORBIDDEN_CALLS:
                findings.append(f"call to {name}() at line {node.lineno}")
        elif (isinstance(node, ast.Constant) and isinstance(node.value, str)
              and id(node) not in docstrings):
            for pattern in ("pkill", "pgrep", "killall"):
                if pattern in node.value:
                    findings.append(f"string mentions {pattern} at line {node.lineno}")
    return findings


class TestModuleHygiene(unittest.TestCase):
    def test_the_module_starts_stops_and_signals_nothing(self):
        """P-AK-SEARCH-1 denial 8 and INC-20260731, as a property of the source.

        A claim module that could signal would be one `pkill` away from taking
        out another session's llama-server. The audit is over the AST, so a
        promise in the docstring is not what is being checked.
        """
        path = Path(crc.__file__)
        findings = _audit_no_process_control(path.read_text(encoding="utf-8"), str(path))
        self.assertEqual(findings, [], f"cpu_region_claim.py is not signal-free: {findings}")

    def test_the_audit_bites(self):
        """Control: the audit must FAIL on code that does what it forbids."""
        bad = "import os, signal\ndef f(pid):\n    os.kill(pid, signal.SIGKILL)\n"
        self.assertTrue(_audit_no_process_control(bad, "<bad>"))
        worse = "def f(out):\n    return 'pkill -f llama-server'\n"
        self.assertTrue(_audit_no_process_control(worse, "<worse>"))

    def test_the_module_imports_no_inference_or_build_machinery(self):
        source = Path(crc.__file__).read_text(encoding="utf-8")
        for banned in ("llama-bench", "llama-server", "cmake", "make -j"):
            self.assertNotIn(banned, source,
                             f"a claim module must not know about {banned}")

    def test_liveness_is_delegated_not_reimplemented(self):
        """One liveness implementation, not two: the GPU sibling's."""
        from autokernel.resource import device_claim as dc
        self.assertIs(crc.assess_holder_liveness, dc.assess_holder_liveness)
        self.assertIs(crc.current_holder_identity, dc.current_holder_identity)

    def test_there_is_no_heartbeat_anywhere(self):
        """INC-20260727: a heartbeat written once is a birth certificate."""
        source = Path(crc.__file__).read_text(encoding="utf-8")
        self.assertNotIn("heartbeat_at", source)
        self.assertNotIn("def heartbeat", source)


class TestAClaimIdIsNotACandidateId(_ClaimTestBase):
    """Two id KINDS, two namespaces, and a validator that can tell them apart.

    Until 2026-08-04 both were spelled `akc-`. A claim id passed where a candidate
    id belongs therefore satisfied the one validator written to catch exactly that
    substitution, and the record grammar rendered `res=akc-…` beside
    `candidate=akc-…` with nothing to say which was which. A shared prefix between
    two id kinds is how a validator becomes decorative.
    """

    @staticmethod
    def _request_with_candidate_id(candidate_id: str):
        """Construct through the REAL validator, with every other field left None.

        `EvaluationRequest.__post_init__` checks the three id prefixes FIRST, so a
        request whose remaining fields are None still reaches — and only reaches —
        the check under test. Resolving against the real class rather than against
        this module's copy of the prefix is the point: the copy is what could be
        wrong.
        """
        from autokernel.evaluator import api  # local: a claim module never imports it
        fields = {name: None for name in api.EvaluationRequest.__dataclass_fields__}
        fields.update(event_id="ake-1", campaign_id="ak-1", candidate_id=candidate_id)
        return api.EvaluationRequest(**fields)

    def test_a_minted_claim_id_is_refused_where_a_candidate_id_belongs(self):
        claim_id = crc._new_id()
        self.assertTrue(claim_id.startswith("akclaim-"), claim_id)
        with self.assertRaises(ValueError) as ctx:
            self._request_with_candidate_id(claim_id)
        self.assertIn("candidate_id", str(ctx.exception))

    def test_a_real_candidate_id_still_passes_that_same_validator(self):
        """Compliant-path control: the guard must not forbid the legitimate id."""
        with self.assertRaises(ValueError) as ctx:
            self._request_with_candidate_id(crc._CANDIDATE_ID_PREFIX + "20260803-0001")
        # It failed on the NEXT field (tier=None), which is the proof that the
        # candidate id itself was accepted.
        self.assertNotIn("candidate_id", str(ctx.exception))
        self.assertIn("tier", str(ctx.exception))

    def test_an_acquired_claims_receipt_carries_the_claim_namespace(self):
        """The id that reaches the record's `res=` field, from a REAL flock."""
        claim = self._acquire()
        try:
            self.assertTrue(claim.claim_id.startswith(crc._CLAIM_ID_PREFIX), claim.claim_id)
            self.assertFalse(claim.claim_id.startswith(crc._CANDIDATE_ID_PREFIX),
                             claim.claim_id)
            self.assertEqual(claim.receipt().claim_id, claim.claim_id)
        finally:
            claim.release()

    def test_the_two_namespaces_cannot_be_re_merged_by_a_later_edit(self):
        """The import-time guard, driven with values it must refuse and must allow."""
        for claim_prefix, candidate_prefix in (("akc-", "akc-"),
                                               ("akc-x", "akc-"),
                                               ("akc-", "akc-x"),
                                               ("", "akc-")):
            with self.subTest(claim=claim_prefix, candidate=candidate_prefix):
                with self.assertRaises(ImportError):
                    crc._require_disjoint_id_namespaces(claim_prefix, candidate_prefix)
        # Compliant path: the constants this module actually ships, and any other
        # genuinely disjoint pair.
        self.assertIsNone(crc._require_disjoint_id_namespaces(
            crc._CLAIM_ID_PREFIX, crc._CANDIDATE_ID_PREFIX))
        self.assertIsNone(crc._require_disjoint_id_namespaces("akclaim-", "akc-"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
