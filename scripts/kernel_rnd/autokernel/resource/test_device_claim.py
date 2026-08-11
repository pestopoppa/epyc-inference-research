#!/usr/bin/env python3
"""Unit tests for device_claim.py — the cross-process exclusive GPU device claim.

NO GPU, NO llama.cpp, NO inference, NO model, NO benchmark. The "device" is a
made-up id (`testdev0`) whose lock file lives in a per-test temp directory, so
nothing here touches the real `/mnt/raid0/llm/tmp` lock root or any device.

WHY THESE TESTS SPAWN REAL PROCESSES
------------------------------------
This is the one module where mocks would prove nothing. The entire claim is
"another PROCESS cannot take this device", and the mechanism is a kernel object
(`flock` on an open file description) plus `/proc` liveness. A mocked lock or a
mocked `/proc` would test the test. So every exclusivity, crash-recovery and
drain assertion here runs against a real `subprocess` child that really acquires
the real lock, and the crash test really SIGKILLs it.

Every child process is one this test created; the test never touches, signals, or
name-pattern-matches any pre-existing process on this shared host. Every child
carries a hard self-timeout so a failed assertion cannot leave a process holding
a lock, and `tearDown` terminates and reaps whatever is left.

Run standalone (no pytest needed):
    python3 scripts/kernel_rnd/autokernel/resource/test_device_claim.py
Or:
    python3 -m unittest scripts/kernel_rnd/autokernel/resource/test_device_claim.py
    python3 -W error::ResourceWarning -m unittest \
        scripts/kernel_rnd/autokernel/resource/test_device_claim.py
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

# Import through the PACKAGE, never by putting this directory on sys.path.
# Two reasons: `autokernel/resource/` would shadow the stdlib `resource` module
# for anything imported afterwards, and AutoPilot's item-12 scar (§2.5) was
# exactly ambient import identity — "which code scores your eval depends on which
# eval ran first in the process".
_KERNEL_RND = str(Path(__file__).resolve().parents[2])
if _KERNEL_RND not in sys.path:
    sys.path.insert(0, _KERNEL_RND)

from autokernel.resource import device_claim as dc  # noqa: E402
from autokernel import schemas as S  # noqa: E402

DEVICE = "testdev0"
OTHER_DEVICE = "testdev1"

# Every child self-terminates after this long no matter what, so a failing test
# cannot leave a lock holder behind on a shared host.
CHILD_MAX_LIFE_S = 45.0

_CHILD_SOURCE = '''\
"""Child worker for test_device_claim.py. Acquires a REAL claim in a REAL process."""
import json, os, sys, time

with open(sys.argv[1]) as _fh:
    cfg = json.load(_fh)
sys.path.insert(0, cfg["kernel_rnd"])
from autokernel.resource import device_claim as dc

deadline = time.time() + cfg["max_life_s"]


def _emit(name, obj):
    tmp = cfg["workdir"] + "/" + name + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(obj, fh)
    os.replace(tmp, cfg["workdir"] + "/" + name)


def _stopped():
    return os.path.exists(cfg["workdir"] + "/stop")


journal = dc.ClaimJournal(cfg["journal"])
mode = cfg["mode"]

if mode == "contend":
    try:
        claim = dc.acquire_device_claim(
            cfg["device"], purpose="contender", campaign_id=cfg["campaign"],
            journal=journal, timeout_s=cfg["timeout_s"], poll_s=0.02,
            stale_grace_s=cfg["stale_grace_s"], lock_root=cfg["lock_root"],
        )
    except dc.DeviceClaimError as exc:
        _emit("result", {"ok": False, "error_type": type(exc).__name__, "error": str(exc)})
        sys.exit(0)
    try:
        _emit("result", {"ok": True, "receipt": claim.receipt().to_dict()})
    finally:
        claim.release()
    sys.exit(0)

claim = dc.acquire_device_claim(
    cfg["device"], purpose=cfg["purpose"], campaign_id=cfg["campaign"],
    journal=journal, timeout_s=cfg["timeout_s"], poll_s=0.02,
    stale_grace_s=cfg["stale_grace_s"], lock_root=cfg["lock_root"],
    max_hold_s=cfg.get("max_hold_s"), holder_label=mode,
)
_emit("ready", {"pid": os.getpid(), "receipt": claim.receipt().to_dict()})

try:
    if mode == "hold_forever":
        # Waits to be SIGKILLed by the parent test. Bounded anyway.
        while time.time() < deadline:
            time.sleep(0.02)
        sys.exit(3)

    if mode == "hold":
        while time.time() < deadline and not _stopped():
            time.sleep(0.02)

    elif mode == "drain_on_revoke":
        # A "task boundary" is one turn of this loop. Nothing preempts the
        # child mid-turn; it looks, acknowledges, finishes, and leaves.
        while time.time() < deadline and not _stopped():
            record = claim.revocation()
            if record is not None:
                claim.acknowledge_revocation()
                _emit("acknowledged", {"at": time.time(),
                                       "revocation_id": record["revocation_id"]})
                time.sleep(0.05)   # the unit of work it was already in
                break
            time.sleep(0.02)

    elif mode == "ignore_revoke":
        # Never asks. Must NOT be preempted; must surface as a defect instead.
        while time.time() < deadline and not _stopped():
            time.sleep(0.02)

    else:
        raise SystemExit("unknown mode " + mode)
finally:
    receipt = claim.release()

_emit("done", {"receipt": receipt.to_dict()})
'''


class _ClaimTestBase(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="device_claim_test_"))
        self.lock_root = self.tmp / "locks"
        self.lock_root.mkdir()
        self.journal_path = self.tmp / "claims.jsonl"
        self.journal = dc.ClaimJournal(self.journal_path)
        self.child_script = self.tmp / "child_worker.py"
        self.child_script.write_text(_CHILD_SOURCE)
        self._children: list[tuple[subprocess.Popen, Path]] = []

    def tearDown(self):
        for _proc, workdir in self._children:
            log = self._child_log(workdir)
            self.assertNotIn("ResourceWarning", log,
                             f"a child leaked a handle in the claim path:\n{log}")
            self.assertNotIn("Exception ignored", log,
                             f"a child raised during cleanup:\n{log}")
        for proc, _workdir in self._children:
            if proc.poll() is None:
                # Only ever a process this test itself created.
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=5)
            else:
                proc.wait()
            self.assertIsNotNone(proc.poll(), "child process was not reaped")
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    # -- child process helpers ------------------------------------------
    def _spawn(self, mode, *, device=DEVICE, timeout_s=10.0, stale_grace_s=0.0,
               purpose="unit-test-hold", campaign="ak-test-20260803",
               max_hold_s=None, name=None):
        workdir = self.tmp / (name or f"child{len(self._children)}")
        workdir.mkdir()
        config = {
            "kernel_rnd": _KERNEL_RND,
            "workdir": str(workdir),
            "journal": str(self.journal_path),
            "lock_root": str(self.lock_root),
            "device": device,
            "mode": mode,
            "purpose": purpose,
            "campaign": campaign,
            "timeout_s": timeout_s,
            "stale_grace_s": stale_grace_s,
            "max_hold_s": max_hold_s,
            "max_life_s": CHILD_MAX_LIFE_S,
        }
        config_path = workdir / "config.json"
        config_path.write_text(json.dumps(config))
        log_path = workdir / "child.log"
        # The parent's copy of the log handle is closed immediately; the child
        # keeps its own dup. Leaving it open would leak a file object into the
        # test and trip -W error::ResourceWarning.
        with open(log_path, "wb") as log:
            # Children run with the ResourceWarning filter armed so a leaked
            # handle inside the claim path shows up in their log. `-W error`
            # alone is NOT a gate: a ResourceWarning raised from `__del__` is
            # printed as "Exception ignored" and the process still exits 0, so
            # `tearDown` asserts on the log text rather than on the exit code.
            proc = subprocess.Popen(
                [sys.executable, "-W", "error::ResourceWarning",
                 str(self.child_script), str(config_path)],
                stdout=log, stderr=log, stdin=subprocess.DEVNULL,
            )
        self._children.append((proc, workdir))
        return proc, workdir

    def _read_json(self, path: Path):
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)

    def _await_file(self, path: Path, timeout_s=15.0, proc=None):
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if path.exists():
                return self._read_json(path)
            if proc is not None and proc.poll() is not None and not path.exists():
                self.fail(f"child exited (rc={proc.returncode}) before writing {path.name}:\n"
                          f"{self._child_log(path.parent)}")
            time.sleep(0.02)
        self.fail(f"timed out waiting for {path}:\n{self._child_log(path.parent)}")

    def _child_log(self, workdir: Path) -> str:
        log = workdir / "child.log"
        if not log.exists():
            return "(no child log)"
        with open(log, "r", encoding="utf-8", errors="replace") as fh:
            return fh.read()

    def _stop_child(self, workdir: Path):
        (workdir / "stop").write_text("1")

    def _await_exit(self, proc, timeout_s=15.0):
        try:
            return proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            self.fail("child did not exit within the drain bound")

    # -- assertions on the journal --------------------------------------
    def _kinds(self):
        return [r["kind"] for r in self.journal.read_all()]

    def _records(self, kind):
        return [r for r in self.journal.read_all() if r["kind"] == kind]

    def _acquire(self, **kwargs):
        params = dict(
            purpose="unit-test", campaign_id="ak-test-20260803",
            journal=self.journal, lock_root=self.lock_root,
            timeout_s=1.0, poll_s=0.02, stale_grace_s=0.0,
        )
        params.update(kwargs)
        device = params.pop("device", DEVICE)
        return dc.acquire_device_claim(device, **params)

    def _write_payload(self, payload, device=DEVICE):
        """Plant a claim payload WITHOUT holding the flock (fault injection)."""
        path = dc.device_lock_path(device, self.lock_root)
        path.write_text(dc.canonical_json(payload) + "\n")
        return path


# =====================================================================
# 1. Cross-process exclusivity
# =====================================================================

class TestCrossProcessExclusivity(_ClaimTestBase):
    def test_second_process_blocks_then_acquires_after_release(self):
        """Two processes contend: the second waits, then gets it on release."""
        holder, holder_dir = self._spawn("hold")
        ready = self._await_file(holder_dir / "ready", proc=holder)
        held_claim_id = ready["receipt"]["claim_id"]

        contender, contender_dir = self._spawn("contend", timeout_s=20.0,
                                               name="contender")
        # Give the contender a moment to be genuinely blocked, then confirm it
        # has NOT acquired while the holder is alive.
        time.sleep(0.4)
        self.assertFalse((contender_dir / "result").exists(),
                         "contender acquired the device while another process held it")
        self.assertIsNone(holder.poll(), "holder exited unexpectedly")

        self._stop_child(holder_dir)
        self.assertEqual(self._await_exit(holder), 0, self._child_log(holder_dir))

        result = self._await_file(contender_dir / "result", proc=contender)
        self.assertTrue(result["ok"], result)
        self.assertNotEqual(result["receipt"]["claim_id"], held_claim_id)
        self.assertEqual(self._await_exit(contender), 0)

    def test_second_process_fails_cleanly_on_timeout(self):
        """A contender that runs out of budget gets DeviceClaimTimeout, not a lock."""
        holder, holder_dir = self._spawn("hold")
        self._await_file(holder_dir / "ready", proc=holder)

        contender, contender_dir = self._spawn("contend", timeout_s=0.5,
                                               name="contender")
        result = self._await_file(contender_dir / "result", proc=contender)
        self.assertFalse(result["ok"])
        self.assertEqual(result["error_type"], "DeviceClaimTimeout")
        # The failure message names the holder, so a human can see who has it.
        self.assertIn("pid=", result["error"])
        self.assertEqual(self._await_exit(contender), 0)

        self._stop_child(holder_dir)
        self.assertEqual(self._await_exit(holder), 0)

    def test_this_process_cannot_take_a_claim_a_child_holds(self):
        holder, holder_dir = self._spawn("hold")
        self._await_file(holder_dir / "ready", proc=holder)
        with self.assertRaises(dc.DeviceClaimTimeout):
            self._acquire(timeout_s=0.4)
        self._stop_child(holder_dir)
        self.assertEqual(self._await_exit(holder), 0)

    def test_second_claim_in_the_same_process_is_refused(self):
        """flock conflicts between open file descriptions, so self-overlap is caught.

        A process-local lease (`gpu_lease.py`) would happily hand the same
        interpreter a second lease; the point of using the kernel's object is
        that it does not.
        """
        claim = self._acquire(timeout_s=1.0)
        try:
            with self.assertRaises(dc.DeviceClaimTimeout):
                self._acquire(timeout_s=0.2)
        finally:
            claim.release()
        second = self._acquire(timeout_s=1.0)
        second.release()


# =====================================================================
# 2. A live holder is never preempted
# =====================================================================

class TestLiveHolderIsNeverPreempted(_ClaimTestBase):
    def test_zero_grace_does_not_preempt_a_live_holder(self):
        """Even with grace=0 — the most aggressive setting — a live holder wins."""
        holder, holder_dir = self._spawn("hold")
        ready = self._await_file(holder_dir / "ready", proc=holder)

        with self.assertRaises(dc.DeviceClaimTimeout):
            self._acquire(timeout_s=0.6, stale_grace_s=0.0)

        self.assertIsNone(holder.poll(), "the holder process was disturbed")
        payload = dc._read_payload_path(dc.device_lock_path(DEVICE, self.lock_root))
        self.assertEqual(payload["claim_id"], ready["receipt"]["claim_id"])
        self.assertEqual(payload["state"], dc.STATE_HELD)
        self.assertNotIn(dc.KIND_RECLAIMED, self._kinds())

        self._stop_child(holder_dir)
        self.assertEqual(self._await_exit(holder), 0)

    def test_live_holder_without_the_lock_is_a_defect_not_a_takeover(self):
        """The ambiguous case: a payload naming a LIVE process, lock free.

        Taking it would be a steal from a possibly-live holder, so the module
        refuses, journals a defect, and leaves the payload untouched.
        """
        planted = {
            "schema": dc.DEVICE_CLAIM_SCHEMA,
            "claim_id": "akd-plantedlive",
            "device_id": DEVICE,
            "state": dc.STATE_HELD,
            "holder": dc.current_holder_identity("planted"),
            "purpose": "fault injection",
            "campaign_id": "ak-test-20260803",
            "acquired_at": dc._utc_now_iso(time.time() - 3600),
            "expires_at": None,
            "reclaimed_from": None,
            "revocation_acknowledged_at": None,
            "revocation_id": None,
        }
        path = self._write_payload(planted)

        with self.assertRaises(dc.DeviceClaimInconsistent) as ctx:
            self._acquire(timeout_s=1.0, stale_grace_s=0.0)
        self.assertIn("alive", str(ctx.exception))

        defects = self._records(dc.KIND_DEFECT)
        self.assertEqual(len(defects), 1)
        self.assertEqual(defects[0]["detail"]["defect_class"],
                         dc.DEFECT_LIVE_HOLDER_FREE_LOCK)
        self.assertEqual(defects[0]["detail"]["liveness"], dc.LIVE)
        self.assertNotIn(dc.KIND_RECLAIMED, self._kinds())
        # Untouched.
        self.assertEqual(dc._read_payload_path(path)["claim_id"], "akd-plantedlive")

    def test_unparseable_payload_is_a_defect_not_a_takeover(self):
        path = dc.device_lock_path(DEVICE, self.lock_root)
        path.write_text("{not json at all")
        with self.assertRaises(dc.DeviceClaimInconsistent):
            self._acquire(timeout_s=1.0, stale_grace_s=0.0)
        defects = self._records(dc.KIND_DEFECT)
        self.assertEqual(defects[0]["detail"]["defect_class"],
                         dc.DEFECT_UNVERIFIABLE_CLAIM)
        self.assertNotIn(dc.KIND_ACQUIRED, self._kinds())

    def test_foreign_host_claim_is_unknown_not_dead(self):
        """A claim from another host cannot be verified here, so it is never reclaimed."""
        holder = dc.current_holder_identity("elsewhere")
        holder["host"] = "some-other-machine"
        verdict = dc.assess_holder_liveness(holder)
        self.assertEqual(verdict.state, dc.UNKNOWN)
        self.assertFalse(verdict.reclaimable)

        planted = {
            "schema": dc.DEVICE_CLAIM_SCHEMA, "claim_id": "akd-foreign",
            "device_id": DEVICE, "state": dc.STATE_HELD, "holder": holder,
            "purpose": "fault injection", "campaign_id": "ak-test",
            "acquired_at": dc._utc_now_iso(time.time() - 86400),
            "expires_at": None, "reclaimed_from": None,
            "revocation_acknowledged_at": None, "revocation_id": None,
        }
        self._write_payload(planted)
        with self.assertRaises(dc.DeviceClaimInconsistent):
            self._acquire(timeout_s=0.5, stale_grace_s=0.0)
        self.assertNotIn(dc.KIND_RECLAIMED, self._kinds())


# =====================================================================
# 3. Crash recovery, journaled
# =====================================================================

class TestCrashRecovery(_ClaimTestBase):
    def _kill_holder(self, proc):
        # SIGKILL a process this test created, to simulate a crash. This is the
        # only kill in the suite and its PID came from our own Popen.
        os.kill(proc.pid, signal.SIGKILL)
        self.assertEqual(proc.wait(timeout=10), -signal.SIGKILL)
        self.assertIsNotNone(proc.poll())

    def test_killed_holder_is_reclaimable_and_the_reclamation_is_journaled(self):
        holder, holder_dir = self._spawn("hold_forever")
        ready = self._await_file(holder_dir / "ready", proc=holder)
        dead_pid = ready["pid"]
        dead_claim_id = ready["receipt"]["claim_id"]

        self._kill_holder(holder)
        # The kernel released the flock at process death; the payload survives.
        payload = dc._read_payload_path(dc.device_lock_path(DEVICE, self.lock_root))
        self.assertEqual(payload["claim_id"], dead_claim_id)

        claim = self._acquire(timeout_s=2.0, stale_grace_s=0.0)
        try:
            receipt = claim.receipt()
            self.assertNotEqual(receipt.claim_id, dead_claim_id)
            self.assertIsNotNone(receipt.reclaimed_from)
            self.assertEqual(receipt.reclaimed_from["claim_id"], dead_claim_id)
            self.assertEqual(receipt.reclaimed_from["holder"]["pid"], dead_pid)
        finally:
            claim.release()

        reclaims = self._records(dc.KIND_RECLAIMED)
        self.assertEqual(len(reclaims), 1)
        detail = reclaims[0]["detail"]
        self.assertEqual(detail["liveness"], dc.DEAD)
        self.assertEqual(detail["reclaimed_from"]["holder"]["pid"], dead_pid)
        self.assertEqual(detail["reclaimed_by_pid"], os.getpid())
        # Journaled BEFORE the takeover: the reclaim record precedes the acquire.
        kinds = self._kinds()
        self.assertLess(kinds.index(dc.KIND_RECLAIMED),
                        len(kinds) - 1 - kinds[::-1].index(dc.KIND_ACQUIRED))

    def test_reclaim_waits_out_the_grace_period(self):
        """The grace period is real: a fresh dead claim is not reclaimed yet."""
        holder, holder_dir = self._spawn("hold_forever")
        self._await_file(holder_dir / "ready", proc=holder)
        self._kill_holder(holder)

        with self.assertRaises(dc.DeviceClaimTimeout) as ctx:
            self._acquire(timeout_s=0.4, stale_grace_s=600.0)
        self.assertIn("grace period", str(ctx.exception))
        self.assertNotIn(dc.KIND_RECLAIMED, self._kinds())

        claim = self._acquire(timeout_s=2.0, stale_grace_s=0.0)
        claim.release()
        self.assertEqual(len(self._records(dc.KIND_RECLAIMED)), 1)

    def test_recycled_pid_cannot_impersonate_a_dead_holder(self):
        """A LIVE pid with the wrong start time is a different process: dead."""
        holder = dc.current_holder_identity("recycled")
        self.assertGreater(holder["start_ticks"], 0)
        holder["start_ticks"] = holder["start_ticks"] + 1   # same pid, other process

        verdict = dc.assess_holder_liveness(holder)
        self.assertEqual(verdict.state, dc.DEAD)
        self.assertIn("recycled", verdict.reason)

        planted = {
            "schema": dc.DEVICE_CLAIM_SCHEMA, "claim_id": "akd-recycled",
            "device_id": DEVICE, "state": dc.STATE_HELD, "holder": holder,
            "purpose": "fault injection", "campaign_id": "ak-test",
            "acquired_at": dc._utc_now_iso(time.time() - 120),
            "expires_at": None, "reclaimed_from": None,
            "revocation_acknowledged_at": None, "revocation_id": None,
        }
        self._write_payload(planted)
        claim = self._acquire(timeout_s=1.0, stale_grace_s=0.0)
        try:
            self.assertEqual(claim.receipt().reclaimed_from["claim_id"], "akd-recycled")
        finally:
            claim.release()
        self.assertEqual(self._records(dc.KIND_RECLAIMED)[0]["detail"]["liveness"], dc.DEAD)

    def test_a_previous_boot_claim_is_dead(self):
        holder = dc.current_holder_identity("old-boot")
        holder["boot_id"] = "00000000-0000-0000-0000-000000000000"
        verdict = dc.assess_holder_liveness(holder)
        self.assertEqual(verdict.state, dc.DEAD)
        self.assertIn("boot", verdict.reason)

    def test_a_zombie_holder_is_dead_not_live(self):
        """An unreaped crashed holder must not look alive forever.

        The zombie's flock was released at exit, so treating its /proc entry as
        liveness would make the device permanently unclaimable.
        """
        proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"],
                                stdin=subprocess.DEVNULL,
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            stat = dc._read_proc_stat(proc.pid)
            self.assertIsNotNone(stat)
            holder = {"pid": proc.pid, "start_ticks": stat[1],
                      "boot_id": dc._read_boot_id(),
                      "host": dc.socket.gethostname(), "label": "zombie-test"}
            self.assertEqual(dc.assess_holder_liveness(holder).state, dc.LIVE)

            proc.send_signal(signal.SIGKILL)
            # Deliberately NOT reaped yet: the process is now a zombie.
            deadline = time.monotonic() + 10.0
            while time.monotonic() < deadline:
                state = dc._read_proc_stat(proc.pid)
                if state is not None and state[0] == "Z":
                    break
                time.sleep(0.02)
            else:
                self.skipTest("could not observe a zombie state on this host")

            verdict = dc.assess_holder_liveness(holder)
            self.assertEqual(verdict.state, dc.DEAD)
            self.assertIn("zombie", verdict.reason)
        finally:
            proc.wait(timeout=10)


# =====================================================================
# 4. Revocation: quiesce and drain
# =====================================================================

class TestRevocation(_ClaimTestBase):
    def test_revoke_transitions_to_revoking_and_the_holder_drains(self):
        holder, holder_dir = self._spawn("drain_on_revoke")
        ready = self._await_file(holder_dir / "ready", proc=holder)
        claim_id = ready["receipt"]["claim_id"]

        record = dc.request_revocation(
            DEVICE, reason="operator needs the card", requested_by="test-operator",
            journal=self.journal, drain_deadline_s=15.0, lock_root=self.lock_root,
        )
        self.assertEqual(record["claim_id"], claim_id)
        self.assertEqual(record["state"], "revoking")

        ack = self._await_file(holder_dir / "acknowledged", proc=holder)
        self.assertEqual(ack["revocation_id"], record["revocation_id"])

        # The holder finishes its unit of work and leaves on its own.
        self.assertEqual(self._await_exit(holder), 0, self._child_log(holder_dir))
        done = self._read_json(holder_dir / "done")
        self.assertEqual(done["receipt"]["claim_id"], claim_id)
        self.assertEqual(done["receipt"]["state"], dc.STATE_DRAINING)
        self.assertIsNotNone(done["receipt"]["released_at"])

        kinds = self._kinds()
        for kind in (dc.KIND_REVOCATION_REQUESTED, dc.KIND_REVOCATION_ACKNOWLEDGED,
                     dc.KIND_REVOCATION_SATISFIED, dc.KIND_RELEASED):
            self.assertIn(kind, kinds)
        self.assertLess(kinds.index(dc.KIND_REVOCATION_ACKNOWLEDGED),
                        kinds.index(dc.KIND_RELEASED))

        verdict = dc.check_revocation_compliance(
            DEVICE, journal=self.journal, lock_root=self.lock_root)
        self.assertEqual(verdict.outcome, dc.PASS)
        self.assertTrue(verdict.passed)
        self.assertEqual(self._records(dc.KIND_DEFECT), [])

        # And the device is immediately claimable again.
        claim = self._acquire(timeout_s=2.0)
        claim.release()

    def test_ignored_revocation_becomes_a_defect_and_nothing_is_killed(self):
        holder, holder_dir = self._spawn("ignore_revoke")
        ready = self._await_file(holder_dir / "ready", proc=holder)

        dc.request_revocation(
            DEVICE, reason="needed elsewhere", requested_by="test-operator",
            journal=self.journal, drain_deadline_s=0.25, lock_root=self.lock_root,
        )
        # Before the deadline the answer is "not decidable yet", never "compliant".
        early = dc.check_revocation_compliance(
            DEVICE, journal=self.journal, lock_root=self.lock_root)
        self.assertEqual(early.outcome, dc.COULD_NOT_CHECK)
        self.assertFalse(early.passed)

        time.sleep(0.4)
        verdict = dc.check_revocation_compliance(
            DEVICE, journal=self.journal, lock_root=self.lock_root)
        self.assertEqual(verdict.outcome, dc.FAIL)
        defects = self._records(dc.KIND_DEFECT)
        self.assertEqual(len(defects), 1)
        self.assertEqual(defects[0]["detail"]["defect_class"], dc.DEFECT_REVOCATION_IGNORED)
        self.assertIsNone(defects[0]["detail"]["acknowledged_at"])

        # The holder is alive and still holds the device: never forcibly taken.
        self.assertIsNone(holder.poll())
        payload = dc._read_payload_path(dc.device_lock_path(DEVICE, self.lock_root))
        self.assertEqual(payload["claim_id"], ready["receipt"]["claim_id"])
        with self.assertRaises(dc.DeviceClaimTimeout):
            self._acquire(timeout_s=0.3, stale_grace_s=0.0)

        self._stop_child(holder_dir)
        self.assertEqual(self._await_exit(holder), 0)

    def test_revocation_requires_a_current_holder(self):
        with self.assertRaises(dc.DeviceClaimError):
            dc.request_revocation(
                DEVICE, reason="nobody home", requested_by="test",
                journal=self.journal, drain_deadline_s=1.0, lock_root=self.lock_root,
            )
        self.assertIsNone(dc.revocation_status(DEVICE, self.lock_root))

    def test_a_new_claim_does_not_inherit_a_predecessors_revocation(self):
        claim = self._acquire(timeout_s=2.0)
        dc.request_revocation(
            DEVICE, reason="drain", requested_by="test", journal=self.journal,
            drain_deadline_s=30.0, lock_root=self.lock_root,
        )
        self.assertIsNotNone(claim.revocation())
        claim.acknowledge_revocation()
        claim.release()

        successor = self._acquire(timeout_s=2.0)
        try:
            self.assertIsNone(successor.revocation(),
                              "a new claim inherited a drain order aimed at its predecessor")
            self.assertIn(dc.KIND_REVOCATION_DISCARDED, self._kinds())
            self.assertIsNone(dc.revocation_status(DEVICE, self.lock_root))
        finally:
            successor.release()

    def test_acknowledging_without_a_revocation_raises(self):
        claim = self._acquire(timeout_s=2.0)
        try:
            with self.assertRaises(dc.DeviceClaimError):
                claim.acknowledge_revocation()
            self.assertNotIn(dc.KIND_REVOCATION_ACKNOWLEDGED, self._kinds())
        finally:
            claim.release()

    def test_compliance_is_could_not_check_without_a_revocation(self):
        claim = self._acquire(timeout_s=2.0)
        try:
            verdict = dc.check_revocation_compliance(
                DEVICE, journal=self.journal, lock_root=self.lock_root)
            self.assertEqual(verdict.outcome, dc.COULD_NOT_CHECK)
            self.assertFalse(verdict.passed)
        finally:
            claim.release()


# =====================================================================
# 5. Receipts
# =====================================================================

class TestReceipt(_ClaimTestBase):
    def test_receipt_round_trips(self):
        claim = self._acquire(timeout_s=2.0, max_hold_s=60.0)
        try:
            receipt = claim.receipt()
            self.assertTrue(receipt.claim_id.startswith("akd-"))
            self.assertEqual(receipt.device_id, DEVICE)
            self.assertEqual(receipt.holder_pid, os.getpid())
            self.assertEqual(receipt.holder_start_ticks, dc._read_proc_stat(os.getpid())[1])
            self.assertEqual(receipt.purpose, "unit-test")
            self.assertEqual(receipt.campaign_id, "ak-test-20260803")
            self.assertIsNone(receipt.released_at)
            self.assertIsNotNone(receipt.expires_at)

            as_dict = receipt.to_dict()
            self.assertEqual(dc.ClaimReceipt.from_dict(as_dict), receipt)
            # Survives a JSON round trip, which is how it reaches an event.
            reloaded = dc.ClaimReceipt.from_dict(json.loads(dc.canonical_json(as_dict)))
            self.assertEqual(reloaded, receipt)
            # Stable bytes regardless of key order.
            shuffled = dict(reversed(list(as_dict.items())))
            self.assertEqual(dc.canonical_json(shuffled), dc.canonical_json(as_dict))
        finally:
            released = claim.release()
        self.assertIsNotNone(released.released_at)
        self.assertEqual(dc.ClaimReceipt.from_dict(released.to_dict()), released)

    def test_receipt_round_trip_rejects_partial_records(self):
        claim = self._acquire(timeout_s=2.0)
        try:
            as_dict = claim.receipt().to_dict()
        finally:
            claim.release()
        for field in ("claim_id", "device_id", "holder_pid", "acquired_at"):
            broken = dict(as_dict)
            broken.pop(field)
            with self.assertRaises(ValueError):
                dc.ClaimReceipt.from_dict(broken)
        with self.assertRaises(ValueError):
            dc.ClaimReceipt.from_dict({**as_dict, "surprise": 1})
        with self.assertRaises(ValueError):
            dc.ClaimReceipt.from_dict({**as_dict, "schema": "something.else.v1"})

    def test_receipt_id_is_accepted_by_the_evaluation_event_schema(self):
        """The receipt id is what binds a measurement to its exclusivity."""
        claim = self._acquire(timeout_s=2.0)
        try:
            event = _minimal_event(claim.receipt().claim_id)
            self.assertEqual(S.validate_evaluation_event(event), [])
            self.assertEqual(event["resource_claim_receipt"], claim.claim_id)
        finally:
            claim.release()

    def test_check_device_claim_held(self):
        claim = self._acquire(timeout_s=2.0)
        try:
            verdict = dc.check_device_claim_held(claim.receipt(), lock_root=self.lock_root)
            self.assertEqual(verdict.outcome, dc.PASS)
            stale = claim.receipt().to_dict()
        finally:
            claim.release()
        # After release the same receipt no longer describes a held device.
        self.assertEqual(
            dc.check_device_claim_held(stale, lock_root=self.lock_root).outcome, dc.FAIL)
        self.assertEqual(
            dc.check_device_claim_held({"device_id": DEVICE}, lock_root=self.lock_root).outcome,
            dc.COULD_NOT_CHECK)


# =====================================================================
# 6. Context-manager behaviour
# =====================================================================

class TestContextManagers(_ClaimTestBase):
    def test_release_on_exception_and_idempotent_release(self):
        with self.assertRaises(ZeroDivisionError):
            with dc.gpu_device_claim(
                DEVICE, purpose="boom", campaign_id="ak-test", journal=self.journal,
                lock_root=self.lock_root, timeout_s=2.0, stale_grace_s=0.0,
            ) as claim:
                self.assertTrue(claim.held)
                1 / 0
        self.assertFalse(claim.held)
        # Released, so the device is immediately available again.
        again = self._acquire(timeout_s=1.0)
        first = again.release()
        # Idempotent: repeated release returns the same receipt and does NOT
        # write a second release record — a duplicate would read as a second
        # hold that never happened.
        self.assertIs(again.release(), first)
        self.assertIs(again.release(), first)
        self.assertEqual(len(self._records(dc.KIND_RELEASED)), 2)
        self.assertEqual(
            [r["detail"]["claim_id"] for r in self._records(dc.KIND_RELEASED)],
            [claim.claim_id, again.claim_id],
        )

    def test_multi_device_claims_release_lifo(self):
        with dc.gpu_device_claims(
            [OTHER_DEVICE, DEVICE], purpose="both", campaign_id="ak-test",
            journal=self.journal, lock_root=self.lock_root, timeout_s=2.0,
            stale_grace_s=0.0,
        ) as claims:
            self.assertEqual(sorted(claims), [DEVICE, OTHER_DEVICE])
            order = [c.device_id for c in claims.values()]
        released = [r["device_id"] for r in self._records(dc.KIND_RELEASED)]
        # Acquired in sorted order; released in reverse.
        self.assertEqual(released, list(reversed(sorted(order))))

    def test_multi_device_is_all_or_nothing(self):
        """A partial acquisition must not leave the first device locked."""
        holder, holder_dir = self._spawn("hold", device=OTHER_DEVICE)
        self._await_file(holder_dir / "ready", proc=holder)

        with self.assertRaises(dc.DeviceClaimTimeout):
            with dc.gpu_device_claims(
                [DEVICE, OTHER_DEVICE], purpose="both", campaign_id="ak-test",
                journal=self.journal, lock_root=self.lock_root, timeout_s=0.4,
                stale_grace_s=0.0,
            ):
                self.fail("acquired a set it could not complete")

        # DEVICE was acquired first and must have been given back.
        free = self._acquire(device=DEVICE, timeout_s=1.0)
        free.release()

        self._stop_child(holder_dir)
        self.assertEqual(self._await_exit(holder), 0)


# =====================================================================
# 7. Fail-closed configuration
# =====================================================================

class TestFailsClosed(_ClaimTestBase):
    def test_journal_is_required(self):
        with self.assertRaises(TypeError):
            dc.acquire_device_claim(DEVICE, purpose="p", campaign_id="c", journal=None,
                                    lock_root=self.lock_root)
        with self.assertRaises(TypeError):
            dc.acquire_device_claim(DEVICE, purpose="p", campaign_id="c",
                                    journal=object(), lock_root=self.lock_root)

    def test_purpose_and_campaign_are_required(self):
        for kwargs in ({"purpose": ""}, {"purpose": "   "}, {"campaign_id": ""}):
            with self.assertRaises(ValueError):
                self._acquire(**kwargs)

    def test_device_id_is_validated_not_sanitized(self):
        for bad in ("", "mi210/0", "../escape", "with space", "x" * 200, None, 7):
            with self.assertRaises(ValueError):
                dc.device_lock_path(bad, self.lock_root)

    def test_bad_tuning_parameters_raise(self):
        with self.assertRaises(ValueError):
            self._acquire(poll_s=0)
        with self.assertRaises(ValueError):
            self._acquire(stale_grace_s=-1)
        with self.assertRaises(ValueError):
            self._acquire(max_hold_s=0)

    def test_zero_timeout_is_one_attempt_not_forever(self):
        holder, holder_dir = self._spawn("hold")
        self._await_file(holder_dir / "ready", proc=holder)
        started = time.monotonic()
        with self.assertRaises(dc.DeviceClaimTimeout):
            self._acquire(timeout_s=0)
        self.assertLess(time.monotonic() - started, 2.0)
        self._stop_child(holder_dir)
        self.assertEqual(self._await_exit(holder), 0)

    def test_a_failed_acquisition_does_not_poison_the_device(self):
        """If journaling the acquire fails, the payload must not survive.

        The lock is dropped on the way out, and a payload naming this LIVE
        process beside a free lock is the one state nothing can resolve
        automatically — it would refuse every later claimant until this process
        exited. The failure must leave the device exactly as it found it.
        """
        class _FailingJournal(dc.ClaimJournal):
            def append(self, kind, device_id, detail):
                if kind == dc.KIND_ACQUIRED:
                    raise OSError("journal filesystem is full")
                return super().append(kind, device_id, detail)

        failing = _FailingJournal(self.journal_path)
        with self.assertRaises(OSError):
            self._acquire(timeout_s=1.0, journal=failing)

        lock_path = dc.device_lock_path(DEVICE, self.lock_root)
        self.assertIsNone(dc._read_payload_path(lock_path),
                          "a failed acquisition left its payload behind")
        # And the device is immediately usable, not permanently refused.
        claim = self._acquire(timeout_s=1.0)
        claim.release()
        self.assertEqual(self._records(dc.KIND_DEFECT), [])

    def test_malformed_journal_line_raises_on_read(self):
        with open(self.journal_path, "a", encoding="utf-8") as fh:
            fh.write("{ this is not json\n")
        with self.assertRaises(dc.DeviceClaimUnreadable):
            self.journal.read_all()

    def test_expiry_is_advisory_and_never_licenses_a_steal(self):
        holder, holder_dir = self._spawn("hold", max_hold_s=0.2)
        self._await_file(holder_dir / "ready", proc=holder)
        self.assertEqual(
            dc.check_claim_expiry(DEVICE, lock_root=self.lock_root).outcome, dc.PASS)
        time.sleep(0.4)
        expired = dc.check_claim_expiry(DEVICE, lock_root=self.lock_root)
        self.assertEqual(expired.outcome, dc.FAIL)
        self.assertIn("still not reclaimable", expired.reasons[0])
        # Expired but alive: still not takeable.
        with self.assertRaises(dc.DeviceClaimTimeout):
            self._acquire(timeout_s=0.3, stale_grace_s=0.0)
        self.assertIsNone(holder.poll())
        self._stop_child(holder_dir)
        self.assertEqual(self._await_exit(holder), 0)

    def test_expiry_without_a_declared_maximum_is_could_not_check(self):
        claim = self._acquire(timeout_s=2.0)
        try:
            verdict = dc.check_claim_expiry(DEVICE, lock_root=self.lock_root)
            self.assertEqual(verdict.outcome, dc.COULD_NOT_CHECK)
        finally:
            claim.release()
        self.assertEqual(
            dc.check_claim_expiry(DEVICE, lock_root=self.lock_root).outcome,
            dc.COULD_NOT_CHECK)


# =====================================================================
# 8. Lock-root sharing with the CPU sibling
# =====================================================================

class TestLockRootSharing(unittest.TestCase):
    """The GPU claim must land in the SAME directory as cpu_region.*.lock.

    Computation only — no file is created in the real lock root.
    """

    def test_default_root_matches_the_cpu_siblings_resolution(self):
        saved = {name: os.environ.pop(name, None) for name in dc._LOCK_ROOT_ENV_VARS}
        try:
            self.assertEqual(
                dc.device_lock_path("mi210_0"),
                Path("/mnt/raid0/llm/tmp/gpu_device.mi210_0.lock"),
            )
            os.environ["ORCHESTRATOR_TMP_DIR"] = "/somewhere/else"
            self.assertEqual(
                dc.device_lock_path("mi210_0"),
                Path("/somewhere/else/gpu_device.mi210_0.lock"),
            )
            del os.environ["ORCHESTRATOR_TMP_DIR"]
            os.environ["ORCHESTRATOR_PATHS_TMP_DIR"] = "/second/choice"
            self.assertEqual(
                dc.device_lock_path("mi210_0"),
                Path("/second/choice/gpu_device.mi210_0.lock"),
            )
        finally:
            for name in dc._LOCK_ROOT_ENV_VARS:
                os.environ.pop(name, None)
                if saved[name] is not None:
                    os.environ[name] = saved[name]

    def test_revocation_path_is_a_sidecar_next_to_the_lock(self):
        root = Path("/tmp/does-not-need-to-exist")
        self.assertEqual(dc.device_lock_path("d0", root).parent,
                         dc.revocation_path("d0", root).parent)
        self.assertNotEqual(dc.device_lock_path("d0", root),
                            dc.revocation_path("d0", root))


# =====================================================================
# 9. Advisory observation is labelled as such
# =====================================================================

class TestInspect(_ClaimTestBase):
    def test_inspect_reports_states_and_labels_itself_advisory(self):
        free = dc.inspect_device_claim(DEVICE, self.lock_root)
        self.assertTrue(free["advisory"])
        self.assertEqual(free["state"], "free")
        self.assertIsNone(free["claim"])

        holder, holder_dir = self._spawn("drain_on_revoke")
        self._await_file(holder_dir / "ready", proc=holder)
        held = dc.inspect_device_claim(DEVICE, self.lock_root)
        self.assertEqual(held["state"], "held")
        self.assertEqual(held["holder_liveness"], dc.LIVE)
        self.assertFalse(held["flock_free"])

        dc.request_revocation(DEVICE, reason="r", requested_by="test",
                              journal=self.journal, drain_deadline_s=15.0,
                              lock_root=self.lock_root)
        self._await_file(holder_dir / "acknowledged", proc=holder)
        revoking = dc.inspect_device_claim(DEVICE, self.lock_root)
        self.assertEqual(revoking["state"], "revoking")
        self.assertIsNotNone(revoking["revocation"])

        self.assertEqual(self._await_exit(holder), 0)

    def test_inspect_reports_a_stale_claim_without_reclaiming_it(self):
        holder, holder_dir = self._spawn("hold_forever")
        self._await_file(holder_dir / "ready", proc=holder)
        os.kill(holder.pid, signal.SIGKILL)
        holder.wait(timeout=10)

        view = dc.inspect_device_claim(DEVICE, self.lock_root)
        self.assertEqual(view["state"], "stale")
        self.assertEqual(view["holder_liveness"], dc.DEAD)
        # Observation changed nothing.
        self.assertIsNotNone(dc._read_payload_path(
            dc.device_lock_path(DEVICE, self.lock_root)))
        self.assertEqual(self._records(dc.KIND_RECLAIMED), [])


# =====================================================================
# 10. Red-team regressions (adversarial review 2026-08-03)
#
# One test per defect that was found by attacking the module rather than by
# exercising it. Each of these FAILED against the module as first written.
# =====================================================================

class _FlakyJournal(dc.ClaimJournal):
    """A journal whose `append` fails for chosen record kinds — the ENOSPC/EIO
    case, which is the only way to reach the module's own durability paths."""

    def __init__(self, path, fail_kinds=()):
        super().__init__(path)
        self.fail_kinds = set(fail_kinds)

    def append(self, kind, device_id, detail):
        if kind in self.fail_kinds:
            raise OSError(28, "No space left on device")
        return super().append(kind, device_id, detail)


class TestRedTeamRegressions(_ClaimTestBase):

    def _planted_payload(self, holder, *, claim_id, age_s=3600.0, device=DEVICE):
        return {
            "schema": dc.DEVICE_CLAIM_SCHEMA, "claim_id": claim_id,
            "device_id": device, "state": dc.STATE_HELD, "holder": holder,
            "purpose": "fault injection", "campaign_id": "ak-test-20260803",
            "acquired_at": dc._utc_now_iso(time.time() - age_s),
            "expires_at": None, "reclaimed_from": None,
            "revocation_acknowledged_at": None, "revocation_id": None,
        }

    # -- D1 -------------------------------------------------------------
    def test_a_holder_without_a_verifiable_boot_id_is_unknown_not_dead(self):
        """A malformed boot id must not be read as "predates the current boot".

        `start_ticks` only means something inside one boot, so a holder with no
        usable boot id is UNVERIFIABLE, not dead. Collapsing it into DEAD made
        the module's headline guarantee false: it reclaimed a claim naming a
        process that was running at that moment.
        """
        for bad in (None, 12345, "", "   ", ["x"]):
            holder = dc.current_holder_identity("no-boot-id")
            if bad is None:
                holder.pop("boot_id")
            else:
                holder["boot_id"] = bad
            verdict = dc.assess_holder_liveness(holder)
            self.assertEqual(verdict.state, dc.UNKNOWN,
                             f"boot_id={bad!r} classified as {verdict.state}: {verdict.reason}")
            self.assertFalse(verdict.reclaimable)

        # End to end: the live process named by the payload is THIS one.
        holder = dc.current_holder_identity("no-boot-id")
        holder.pop("boot_id")
        self._write_payload(self._planted_payload(holder, claim_id="akd-liveNoBootId"))

        with self.assertRaises(dc.DeviceClaimInconsistent):
            self._acquire(timeout_s=0.5, stale_grace_s=0.0)
        self.assertNotIn(dc.KIND_RECLAIMED, self._kinds())
        self.assertEqual(self._records(dc.KIND_DEFECT)[0]["detail"]["liveness"], dc.UNKNOWN)
        self.assertEqual(
            dc._read_payload_path(dc.device_lock_path(DEVICE, self.lock_root))["claim_id"],
            "akd-liveNoBootId", "the live holder's payload was modified")

    # -- D2 -------------------------------------------------------------
    def test_truncating_the_lock_file_cannot_turn_an_ignored_revocation_into_a_pass(self):
        """The standing screen: can this check be passed by deleting what it
        inspects? The lock root is a shared, world-writable directory, so
        "no payload" must be corroborated by a FREE lock before it counts as a
        release — otherwise `truncate -s 0` launders an ignored revocation."""
        holder, holder_dir = self._spawn("ignore_revoke")
        self._await_file(holder_dir / "ready", proc=holder)
        dc.request_revocation(DEVICE, reason="need the device", requested_by="test",
                              journal=self.journal, drain_deadline_s=0.0,
                              lock_root=self.lock_root)
        time.sleep(0.1)
        self.assertEqual(
            dc.check_revocation_compliance(DEVICE, journal=self.journal,
                                           lock_root=self.lock_root).outcome, S.FAIL)

        lock_path = dc.device_lock_path(DEVICE, self.lock_root)
        os.truncate(lock_path, 0)          # holder is untouched and still has LOCK_EX
        verdict = dc.check_revocation_compliance(DEVICE, journal=self.journal,
                                                 lock_root=self.lock_root)
        self.assertEqual(verdict.outcome, S.COULD_NOT_CHECK)
        self.assertFalse(verdict.passed)
        self.assertIn("STILL HELD", verdict.reasons[0])
        self.assertIsNone(holder.poll(), "the holder was disturbed")
        self.assertFalse(dc._probe_lock_free(lock_path))

        self._stop_child(holder_dir)
        self.assertEqual(self._await_exit(holder), 0)
        # Once the holder really lets go, the same absence IS a release.
        self.assertEqual(
            dc.check_revocation_compliance(DEVICE, journal=self.journal,
                                           lock_root=self.lock_root).outcome, S.PASS)

    # -- D4 -------------------------------------------------------------
    def test_a_dead_holder_is_not_accused_of_ignoring_the_revocation(self):
        """A holder that died did not ignore a drain order. Filing it as
        `revocation_ignored` sends a human to escalate with the owner of a
        process that no longer exists."""
        holder, holder_dir = self._spawn("hold_forever")
        self._await_file(holder_dir / "ready", proc=holder)
        dc.request_revocation(DEVICE, reason="need the device", requested_by="test",
                              journal=self.journal, drain_deadline_s=0.0,
                              lock_root=self.lock_root)
        os.kill(holder.pid, signal.SIGKILL)     # a PID this test itself created
        holder.wait(timeout=10)
        time.sleep(0.1)

        verdict = dc.check_revocation_compliance(DEVICE, journal=self.journal,
                                                 lock_root=self.lock_root)
        self.assertEqual(verdict.outcome, S.FAIL)
        detail = self._records(dc.KIND_DEFECT)[-1]["detail"]
        self.assertEqual(detail["defect_class"], dc.DEFECT_REVOCATION_ORPHANED)
        self.assertNotEqual(detail["defect_class"], dc.DEFECT_REVOCATION_IGNORED)
        self.assertEqual(detail["holder_liveness"], dc.DEAD)
        self.assertNotIn("escalate", detail["note"])

    # -- D5 -------------------------------------------------------------
    def test_re_asking_for_a_revocation_does_not_buy_the_holder_more_time(self):
        """A revoker that nudges more often than its own drain bound must not
        push the deadline out forever — that would make the ignored-revocation
        defect unreachable by the act of asking again."""
        holder, holder_dir = self._spawn("ignore_revoke")
        self._await_file(holder_dir / "ready", proc=holder)
        first = dc.request_revocation(DEVICE, reason="first", requested_by="test",
                                      journal=self.journal, drain_deadline_s=0.3,
                                      lock_root=self.lock_root)
        time.sleep(0.4)
        again = dc.request_revocation(DEVICE, reason="nudge", requested_by="test",
                                      journal=self.journal, drain_deadline_s=0.3,
                                      lock_root=self.lock_root)
        self.assertEqual(again["drain_deadline_at"], first["drain_deadline_at"],
                         "the re-request moved the drain deadline")
        self.assertEqual(again["first_requested_at"], first["first_requested_at"])
        self.assertEqual(again["supersedes"], first["revocation_id"])
        self.assertEqual(
            dc.check_revocation_compliance(DEVICE, journal=self.journal,
                                           lock_root=self.lock_root).outcome, S.FAIL)
        self.assertIsNone(holder.poll(), "nothing may preempt the holder")
        self._stop_child(holder_dir)
        self.assertEqual(self._await_exit(holder), 0)

    # -- D6 -------------------------------------------------------------
    def test_an_oversize_payload_is_corruption_to_both_readers(self):
        """The unlocked reader's `read()` is bounded, so without an explicit
        size check it parses a TRUNCATED PREFIX of a file that the locked reader
        rejects outright — two readers of one file disagreeing about whether it
        is a claim."""
        holder = dc.current_holder_identity("oversize")
        payload = self._planted_payload(holder, claim_id="akd-oversize")
        path = dc.device_lock_path(DEVICE, self.lock_root)
        path.write_text(dc.canonical_json(payload) + " " * (dc._MAX_PAYLOAD_BYTES + 8) + "\n")
        self.assertGreater(path.stat().st_size, dc._MAX_PAYLOAD_BYTES)

        with self.assertRaises(dc.DeviceClaimUnreadable):
            dc._read_payload_path(path)
        fd = os.open(path, os.O_RDWR)
        try:
            with self.assertRaises(dc.DeviceClaimUnreadable):
                dc._read_payload_fd(fd)
        finally:
            os.close(fd)
        # ...and the acquire path treats it as a defect, never as a free device.
        with self.assertRaises(dc.DeviceClaimInconsistent):
            self._acquire(timeout_s=0.3, stale_grace_s=0.0)
        self.assertEqual(self._records(dc.KIND_DEFECT)[0]["detail"]["defect_class"],
                         dc.DEFECT_UNVERIFIABLE_CLAIM)

    def test_an_oversize_revocation_record_is_corruption(self):
        path = dc.revocation_path(DEVICE, self.lock_root)
        path.write_text('{"schema":"x"}' + " " * (dc._MAX_PAYLOAD_BYTES + 8))
        with self.assertRaises(dc.DeviceClaimUnreadable):
            dc.revocation_status(DEVICE, self.lock_root)

    # -- R1 -------------------------------------------------------------
    def test_a_failed_release_journal_write_is_retried_not_latched(self):
        """The lock is gone once release returns, so nothing can reconstruct the
        release afterwards. Caching "already released" over a failed record
        write dropped the outcome permanently (invariant 7)."""
        journal = _FlakyJournal(self.tmp / "flaky.jsonl", {dc.KIND_RELEASED})
        claim = self._acquire(journal=journal)
        with self.assertRaises(OSError):
            claim.release()
        self.assertFalse(claim.held)
        self.assertTrue(dc._probe_lock_free(dc.device_lock_path(DEVICE, self.lock_root)),
                        "the lock must be released even when journaling fails")
        kinds = [r["kind"] for r in journal.read_all()]
        self.assertNotIn(dc.KIND_RELEASED, kinds)

        journal.fail_kinds.clear()          # the transient fault clears
        receipt = claim.release()           # idempotent call retries the record
        kinds = [r["kind"] for r in journal.read_all()]
        self.assertEqual(kinds.count(dc.KIND_RELEASED), 1)
        self.assertEqual(receipt.claim_id, claim.claim_id)
        claim.release()                     # and does not duplicate it
        self.assertEqual([r["kind"] for r in journal.read_all()].count(dc.KIND_RELEASED), 1)

    # -- R2 -------------------------------------------------------------
    def test_a_failed_payload_clear_on_release_is_journaled_as_a_defect(self):
        """The mirror image of the acquire-path poisoning bug. If the payload
        cannot be cleared, the lock still drops and the device is left naming a
        LIVE process beside a free lock — unresolvable by any machine. That
        cannot be repaired here, so it must be loud and durable, never silent."""
        claim = self._acquire()
        original = dc._clear_payload

        def _boom(fd):
            raise OSError(5, "Input/output error")

        dc._clear_payload = _boom
        try:
            with self.assertRaises(dc.DeviceClaimError) as ctx:
                claim.release()
        finally:
            dc._clear_payload = original
        self.assertIn("NOT claimable", str(ctx.exception))

        defects = self._records(dc.KIND_DEFECT)
        self.assertEqual(len(defects), 1)
        self.assertEqual(defects[0]["detail"]["defect_class"],
                         dc.DEFECT_LIVE_HOLDER_FREE_LOCK)
        self.assertEqual(defects[0]["detail"]["claim_id"], claim.claim_id)
        released = self._records(dc.KIND_RELEASED)
        self.assertEqual(len(released), 1)
        self.assertIsNotNone(released[0]["detail"]["payload_clear_error"])
        # The state really is poisoned — the defect record is the only exit.
        with self.assertRaises(dc.DeviceClaimInconsistent):
            self._acquire(timeout_s=0.3, stale_grace_s=0.0)

    # -- R3 -------------------------------------------------------------
    def test_multi_device_release_unwinds_every_claim_when_one_release_raises(self):
        """`gpu_device_claims` is all-or-nothing on the way in; it must be
        all-or-nothing on the way out too. A release that raises left the
        sibling locks held by this process, and nothing else can free them."""
        journal = _FlakyJournal(self.tmp / "flaky2.jsonl")
        with self.assertRaises(OSError):
            with dc.gpu_device_claims(
                [DEVICE, OTHER_DEVICE], purpose="unit-test",
                campaign_id="ak-test-20260803", journal=journal,
                timeout_s=1.0, poll_s=0.02, stale_grace_s=0.0,
                lock_root=self.lock_root,
            ):
                journal.fail_kinds.add(dc.KIND_RELEASED)
        journal.fail_kinds.clear()
        for device in (DEVICE, OTHER_DEVICE):
            self.assertTrue(
                dc._probe_lock_free(dc.device_lock_path(device, self.lock_root)),
                f"{device} was left locked by a failed sibling release")


def _sha(text: str) -> str:
    import hashlib
    return hashlib.sha256(text.encode()).hexdigest()


def _minimal_event(receipt_id: str) -> dict:
    """A valid evaluation_event whose resource_claim_receipt is a real claim id."""
    return {
        "schema": S.SCHEMA_EVALUATION_EVENT,
        "event_id": "ake-20260803-0001",
        "campaign_id": "ak-test-20260803",
        "candidate_id": "akc-20260803-0001",
        "tier": "T1",
        "change_class": "parameter", "anchor_tier": "T1",
        "transfer_ratio_to": [], "backend": "llama_gpu",
        "device_state": {
            "device_id": "mi210_0", "source": "fixture/rocm-smi",
            "nominal_sclk_mhz": 1700.0, "min_sclk_ratio": 0.9,
            "samples": [{"sclk_mhz": 1700.0, "mclk_mhz": 1600.0,
                         "power_w": 180.0, "temperature_c": 55.0,
                         "under_measurement_load": True}],
            "throttle_observed": False,
            "receipt_ref": "fixture://device-state/device-claim",
        },
        "claim_grammar": {
            "category": "CANDIDATE",
            "protocol_id": "P-AK-SEARCH-1/v1",
            "metric": "decode_tokens_per_s",
            "metric_direction": "higher_better",
            "reps": 5,
            "attestation_ref": "rcpt-host-20260803T101500Z",
        },
        "evaluator": {"id": "P-AK-SEARCH-1/v1", "bundle_sha256": _sha("bundle")},
        "artifact": {
            "source_sha256": _sha("snapshot"),
            "binary_sha256": _sha("candidate-binary"),
            "linkage_sha256": _sha("candidate-linkage"),
        },
        "anchor": {
            "source_commit": "67a433bf45a8a091d83b4ea0b32ff0735fd51800",
            "binary_sha256": _sha("anchor-binary"),
            "linkage_sha256": _sha("anchor-linkage"),
            "measurement_event_ids": ["ake-20260801-0009"],
        },
        "scope_manifest_sha256": _sha("scope-manifest"),
        "host_receipt": "rcpt-host-20260803T101500Z",
        "resource_claim_receipt": receipt_id,
        "co_residency": "single",
        "correctness": {"test_backend_ops": "pass"},
        "quality": {},
        "stability": {},
        "scope_denominator": {
            "machine_subset": "partial",
            "numa_nodes": [0],
            "devices": ["gfx90a:0"],
            "cores": 8,
        },
        "determinism": {"class": "bitwise_stable", "same_seed_repeat_runs": 3},
        "performance": {
            "raw_samples": [51.2, 51.4, 51.1],
            "paired_blocks": 3,
            "estimate": 51.23,
            "uncertainty": {"e_process_value": 12.4},
        },
        "mechanism": {},
        "integrity_flags": [],
        "status": "pass",
        "supersedes": [],
        "created_at": "2026-08-03T10:45:00+00:00",
    }


if __name__ == "__main__":
    unittest.main(verbosity=2)
