"""Unit tests for autokernel/resource/preflight.py — the single audited,
read-only inference preflight (§3.5).

NO inference, NO server, NO model, NO GPU, and — the point of the module —
NO process is started, stopped, or signalled by anything here. Every process
tree in this suite is a directory of text files shaped like `/proc`, which is
also why the suite is safe to run on the shared host at any time: it never
observes, and never depends on, another session's live processes.

The suite is organised around the failures the module exists to prevent:

  * **signalling is structurally absent** — the module's own AST audit passes,
    an INDEPENDENT token-level scan of the source agrees, and the audit is
    proved non-vacuous by making it FAIL on synthetic sources that do signal;
  * **three verdicts stay three** — PASS / FAIL / COULD_NOT_CHECK are distinct,
    truth-testing the result raises, and FAIL and COULD_NOT_CHECK raise
    DIFFERENT exception types so one `except` cannot merge them;
  * **claim witness is preferred** — when claims are readable the interim
    name-pattern scan is never even invoked (asserted with a scanner that
    raises if called);
  * **an unreadable claim root is COULD_NOT_CHECK, never a pass** — missing,
    non-directory, empty, and permission-denied roots all land there;
  * **the earlyoom false positive cannot recur** — `earlyoom --ignore
    '^(llama-server|sd-server)$'` (INC-20260731) is classified as a guard's
    exclusion list, not as inference.

Run standalone (no pytest needed):
    python3 -m unittest scripts/kernel_rnd/autokernel/resource/test_preflight.py
    python3 scripts/kernel_rnd/autokernel/resource/test_preflight.py
"""
from __future__ import annotations

import io
import json
import os
import sys
import tempfile
import token as token_module
import tokenize
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import preflight as P  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import schemas as S  # noqa: E402


# ---------------------------------------------------------------------------
# Fixture helpers: a `/proc` made of text files, and real lock files whose
# dev:inode is fed into a synthetic /proc/locks. The lock files are REAL so the
# dev:inode matching code under test is exercised for real; only the holder
# table is synthetic, because creating a real second holder would mean starting
# a process.
# ---------------------------------------------------------------------------

def _write_stat(proc_root: Path, pid: int, ppid: int, comm: str, starttime: int = 4242) -> None:
    fields = ["S", str(ppid)] + ["0"] * 17 + [str(starttime)] + ["0"] * 12
    (proc_root / str(pid)).mkdir(parents=True, exist_ok=True)
    (proc_root / str(pid) / "stat").write_text(f"{pid} ({comm}) " + " ".join(fields) + "\n")


def _mkproc(
    proc_root: Path,
    pid: int,
    *,
    ppid: int = 1,
    comm: str = "python3",
    argv=("/usr/bin/python3", "runner.py"),
    cgroup: str = "0::/agent.slice/session-1.scope",
    starttime: int = 4242,
) -> None:
    """Create one fake /proc/<pid> entry."""
    _write_stat(proc_root, pid, ppid, comm, starttime)
    pid_dir = proc_root / str(pid)
    (pid_dir / "cmdline").write_bytes(b"".join(a.encode() + b"\0" for a in argv))
    (pid_dir / "cgroup").write_text(f"{cgroup}\n")


def _locks_line(index: int, path: Path, pid: int, klass: str = "FLOCK",
                blocked: bool = False) -> str:
    info = os.stat(path)
    dev = "%02x:%02x" % (os.major(info.st_dev), os.minor(info.st_dev))
    arrow = "-> " if blocked else ""
    return f"{index}: {arrow}{klass}  ADVISORY  WRITE {pid} {dev}:{info.st_ino} 0 EOF"


_PAYLOAD_KEYS = ("schema_version", "pid", "role", "region", "regions",
                 "instance_idx", "request_tag", "started_at")


def _payload(pid: int, role: str, region: str, tag: str = "bench-canonical") -> str:
    return json.dumps({
        "schema_version": 1,
        "pid": pid,
        "role": role,
        "region": region,
        "regions": [region],
        "instance_idx": 0,
        "request_tag": tag,
        "started_at": 1754179200.0,
    }, sort_keys=True)


class _Fixture(unittest.TestCase):
    """A temp dir holding a fake /proc and a fake region-lock namespace."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.proc_root = self.root / "proc"
        self.lock_dir = self.root / "tmp"
        self.proc_root.mkdir()
        self.lock_dir.mkdir()
        self.self_pid = 5000
        _mkproc(self.proc_root, 1, ppid=0, comm="init", argv=("/sbin/init",))
        _mkproc(self.proc_root, self.self_pid, ppid=1, comm="python3")
        self._lock_lines: list = []
        self._write_locks()

    # -- fake /proc/locks ---------------------------------------------------
    def _write_locks(self) -> None:
        (self.proc_root / "locks").write_text("\n".join(self._lock_lines) + "\n")

    def add_lock(self, role: str, region: str, *, holder=None, payload: str = "",
                 blocked_waiter=None, klass: str = "FLOCK") -> Path:
        path = self.lock_dir / f"cpu_region.{role}.{region}.lock"
        path.write_text(payload)
        index = len(self._lock_lines) + 1
        if holder is not None:
            self._lock_lines.append(_locks_line(index, path, holder, klass=klass))
        if blocked_waiter is not None:
            self._lock_lines.append(
                _locks_line(index + 1, path, blocked_waiter, klass=klass, blocked=True))
        self._write_locks()
        return path

    # -- convenience accessors ---------------------------------------------
    def proc(self) -> "P.ProcSource":
        return P.ProcSource(root=self.proc_root, self_pid=self.self_pid)

    def sources(self, **kwargs) -> "P.ClaimSources":
        kwargs.setdefault("region_lock_dir", self.lock_dir)
        kwargs.setdefault("proc", self.proc())
        return P.ClaimSources(**kwargs)

    def scope(self, **kwargs) -> "P.PreflightScope":
        kwargs.setdefault("label", "unit-test-bench")
        return P.PreflightScope(**kwargs)


# ===========================================================================
# 1. Signalling is structurally absent
# ===========================================================================

class SignallingAbsenceTest(unittest.TestCase):

    def test_module_audits_clean(self):
        check = P.audit_no_signalling_capability()
        self.assertEqual(check.outcome, P.PASS, msg=f"violations: {check.reasons}")

    def test_audit_target_is_this_module(self):
        self.assertEqual(P.SIGNALLING_AUDIT_TARGET, Path(P.__file__).resolve())

    def test_independent_token_scan_finds_no_forbidden_identifier(self):
        """An INDEPENDENT check, not a re-run of the module's own auditor.

        Tokenises the source and looks at NAME tokens only, so comments,
        docstrings and f-string literal text are excluded by construction. If
        the module's AST audit were subverted this would still catch a literal
        `os.kill` in executable code.
        """
        source = P.SIGNALLING_AUDIT_TARGET.read_text(encoding="utf-8")
        names = {
            tok.string
            for tok in tokenize.generate_tokens(io.StringIO(source).readline)
            if tok.type == token_module.NAME
        }
        forbidden = {"kill", "killpg", "send_signal", "terminate", "raise_signal",
                     "pthread_kill", "system", "popen", "fork", "abort", "execv",
                     "subprocess", "signal", "psutil", "ctypes", "getattr",
                     "SIGKILL", "SIGTERM"}
        self.assertEqual(names & forbidden, set())

    def test_no_forbidden_module_is_imported_at_runtime(self):
        """Belt and braces: the loaded module object holds no such reference."""
        for name in ("subprocess", "signal", "ctypes", "psutil"):
            self.assertNotIn(name, vars(P))

    # -- the audit is not vacuous ------------------------------------------
    def _audit_source(self, body: str):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        path = Path(tmp.name) / "candidate.py"
        path.write_text(body, encoding="utf-8")
        return P.audit_no_signalling_capability(path)

    def test_audit_fails_on_os_kill(self):
        check = self._audit_source("import os\ndef stop(pid):\n    os.kill(pid, 9)\n")
        self.assertEqual(check.outcome, P.FAIL)
        self.assertTrue(any("kill" in r for r in check.reasons))

    def test_audit_fails_on_signal_import(self):
        check = self._audit_source("import signal\n")
        self.assertEqual(check.outcome, P.FAIL)

    def test_audit_fails_on_from_signal_import(self):
        check = self._audit_source("from signal import SIGKILL\n")
        self.assertEqual(check.outcome, P.FAIL)

    def test_audit_fails_on_subprocess_shellout(self):
        check = self._audit_source("import subprocess\nsubprocess.run(['pkill', 'llama'])\n")
        self.assertEqual(check.outcome, P.FAIL)

    def test_audit_fails_on_dynamic_escape_hatch(self):
        check = self._audit_source("import os\nf = getattr(os, 'k' + 'ill')\n")
        self.assertEqual(check.outcome, P.FAIL)

    def test_audit_passes_on_a_docstring_that_merely_discusses_signalling(self):
        """Prose about pkill must not be mistaken for a call — this module's own
        docstring is full of it."""
        check = self._audit_source('"""Never os.kill; never pkill; SIGKILL is banned."""\nx = 1\n')
        self.assertEqual(check.outcome, P.PASS)

    # -- inability to evaluate is the third outcome ------------------------
    def test_audit_could_not_check_when_source_missing(self):
        check = P.audit_no_signalling_capability(Path("/nonexistent/preflight.py"))
        self.assertEqual(check.outcome, P.COULD_NOT_CHECK)
        self.assertFalse(check.passed)

    def test_audit_could_not_check_when_source_unparseable(self):
        check = self._audit_source("def broken(:\n")
        self.assertEqual(check.outcome, P.COULD_NOT_CHECK)


# ===========================================================================
# 2. The three verdicts are distinct and cannot be collapsed
# ===========================================================================

class VerdictTest(unittest.TestCase):

    def _result(self, verdict, **kwargs):
        kwargs.setdefault("basis", P.BASIS_CLAIM_WITNESS)
        kwargs.setdefault("scope", P.PreflightScope.whole_machine_cpu("t"))
        kwargs.setdefault("observed_at", "2026-08-03T00:00:00Z")
        if verdict == P.FAIL:
            kwargs.setdefault("findings", (P.Finding("process", "llama-server", "pid 7"),))
        if verdict == P.COULD_NOT_CHECK:
            kwargs.setdefault("reasons", ("claim root unreadable",))
        return P.PreflightResult(verdict=verdict, **kwargs)

    def test_three_verdicts_are_distinct_strings(self):
        self.assertEqual(len({P.PASS, P.FAIL, P.COULD_NOT_CHECK}), 3)

    def test_passed_is_true_only_for_pass(self):
        self.assertTrue(self._result(P.PASS).passed)
        self.assertFalse(self._result(P.FAIL).passed)
        self.assertFalse(self._result(P.COULD_NOT_CHECK).passed)

    def test_could_not_check_flag_does_not_overlap_pass(self):
        self.assertTrue(self._result(P.COULD_NOT_CHECK).could_not_check)
        self.assertFalse(self._result(P.PASS).could_not_check)
        self.assertFalse(self._result(P.FAIL).could_not_check)

    def test_truth_testing_raises(self):
        for verdict in (P.PASS, P.FAIL, P.COULD_NOT_CHECK):
            with self.assertRaises(TypeError):
                bool(self._result(verdict))

    def test_truth_testing_raises_in_an_if_statement(self):
        result = self._result(P.COULD_NOT_CHECK)
        with self.assertRaises(TypeError):
            if result:  # noqa: F841 - the point is that this line cannot run
                pass

    def test_require_pass_returns_self_on_pass(self):
        result = self._result(P.PASS)
        self.assertIs(result.require_pass(), result)

    def test_require_pass_raises_distinct_types(self):
        with self.assertRaises(P.ConcurrentInferenceDetected):
            self._result(P.FAIL).require_pass()
        with self.assertRaises(P.PreflightIndeterminate):
            self._result(P.COULD_NOT_CHECK).require_pass()

    def test_could_not_check_is_not_catchable_as_concurrent_inference(self):
        """The two failure modes must not be mergeable by one except clause."""
        with self.assertRaises(P.PreflightIndeterminate):
            try:
                self._result(P.COULD_NOT_CHECK).require_pass()
            except P.ConcurrentInferenceDetected:
                self.fail("COULD_NOT_CHECK was caught as ConcurrentInferenceDetected")

    def test_both_failure_types_share_one_base(self):
        for exc in (P.ConcurrentInferenceDetected, P.PreflightIndeterminate):
            self.assertTrue(issubclass(exc, P.PreflightNotSatisfied))

    def test_fail_without_a_finding_is_refused(self):
        with self.assertRaises(ValueError):
            P.PreflightResult(verdict=P.FAIL, basis=P.BASIS_CLAIM_WITNESS,
                              scope=P.PreflightScope.whole_machine_cpu("t"),
                              observed_at="2026-08-03T00:00:00Z")

    def test_could_not_check_without_a_reason_is_refused(self):
        with self.assertRaises(ValueError):
            P.PreflightResult(verdict=P.COULD_NOT_CHECK, basis=P.BASIS_CLAIM_WITNESS,
                              scope=P.PreflightScope.whole_machine_cpu("t"),
                              observed_at="2026-08-03T00:00:00Z")

    def test_invalid_verdict_refused(self):
        with self.assertRaises(ValueError):
            P.PreflightResult(verdict="OK", basis=P.BASIS_CLAIM_WITNESS,
                              scope=P.PreflightScope.whole_machine_cpu("t"),
                              observed_at="2026-08-03T00:00:00Z")

    def test_invalid_basis_refused(self):
        with self.assertRaises(ValueError):
            P.PreflightResult(verdict=P.PASS, basis="VIBES",
                              scope=P.PreflightScope.whole_machine_cpu("t"),
                              observed_at="2026-08-03T00:00:00Z")

    def test_as_check_round_trips_into_the_schemas_vocabulary(self):
        for verdict in (P.PASS, P.FAIL, P.COULD_NOT_CHECK):
            check = self._result(verdict).as_check()
            self.assertIsInstance(check, S.Check)
            self.assertEqual(check.outcome, verdict)
            self.assertEqual(check.passed, verdict == P.PASS)

    def test_combine_verdicts_lattice(self):
        self.assertEqual(P.combine_verdicts(P.PASS, P.PASS), P.PASS)
        self.assertEqual(P.combine_verdicts(P.PASS, P.COULD_NOT_CHECK), P.COULD_NOT_CHECK)
        self.assertEqual(P.combine_verdicts(P.COULD_NOT_CHECK, P.FAIL), P.FAIL)
        self.assertEqual(P.combine_verdicts(P.FAIL, P.PASS), P.FAIL)

    def test_combine_verdicts_refuses_garbage_and_emptiness(self):
        with self.assertRaises(ValueError):
            P.combine_verdicts()
        with self.assertRaises(ValueError):
            P.combine_verdicts(P.PASS, "MAYBE")


# ===========================================================================
# 3. Scope
# ===========================================================================

class ScopeTest(unittest.TestCase):

    def test_empty_scope_is_refused(self):
        with self.assertRaises(ValueError):
            P.PreflightScope(label="nothing", cpu_regions=frozenset())

    def test_none_cpu_regions_means_whole_machine(self):
        scope = P.PreflightScope.whole_machine_cpu("bench")
        self.assertTrue(scope.covers_region("q0"))
        self.assertTrue(scope.covers_region("anything"))
        self.assertTrue(scope.covers_cpu)

    def test_named_regions_do_not_cover_others(self):
        scope = P.PreflightScope.cpu("bench", ["q0", "q1"])
        self.assertTrue(scope.covers_region("q0"))
        self.assertFalse(scope.covers_region("q2"))

    def test_gpu_only_scope_does_not_cover_cpu(self):
        scope = P.PreflightScope.gpu("gpu-bench", ["gfx90a:0"])
        self.assertFalse(scope.covers_cpu)

    def test_blank_label_refused(self):
        with self.assertRaises(ValueError):
            P.PreflightScope(label="   ")

    def test_scope_dict_is_canonicalisable(self):
        S.canonical_json(P.PreflightScope.whole_machine_cpu("bench").to_dict())


# ===========================================================================
# 4. Owned-scope enumeration
# ===========================================================================

class OwnedScopeTest(_Fixture):

    def test_self_ancestors_and_descendants_are_owned(self):
        # 1 -> 4000 (region-lock wrapper) -> 5000 (us) -> 6000 (child)
        _mkproc(self.proc_root, 4000, ppid=1, comm="region-lock")
        _write_stat(self.proc_root, self.self_pid, 4000, "python3")
        _mkproc(self.proc_root, 6000, ppid=self.self_pid, comm="llama-bench")
        owned = P.read_own_scope(self.proc())
        self.assertEqual(owned.pids, frozenset({4000, self.self_pid, 6000}))
        self.assertEqual(owned.reasons[4000], "ancestor")
        self.assertEqual(owned.reasons[6000], "descendant")
        self.assertEqual(owned.reasons[self.self_pid], "self")

    def test_pid_1_is_never_owned(self):
        owned = P.read_own_scope(self.proc())
        self.assertNotIn(1, owned.pids)

    def test_unrelated_process_is_not_owned(self):
        _mkproc(self.proc_root, 7000, ppid=1, comm="llama-server")
        owned = P.read_own_scope(self.proc())
        self.assertFalse(owned.owns(7000))

    def test_cgroup_is_recorded(self):
        owned = P.read_own_scope(self.proc())
        self.assertEqual(owned.cgroup, "/agent.slice/session-1.scope")

    def test_missing_proc_root_raises_rather_than_returning_empty(self):
        proc = P.ProcSource(root=self.root / "no-such-proc", self_pid=self.self_pid)
        with self.assertRaises(P.PreflightUnavailable):
            P.read_own_scope(proc)

    def test_owned_scope_dict_is_canonicalisable(self):
        S.canonical_json(P.read_own_scope(self.proc()).to_dict())


# ===========================================================================
# 5. Claim witness — CPU region locks
# ===========================================================================

class RegionClaimTest(_Fixture):

    def test_foreign_holder_fails_with_what_and_whose(self):
        _mkproc(self.proc_root, 9100, ppid=1, comm="llama-server",
                argv=("/mnt/raid0/llm/llama.cpp/build/bin/llama-server", "-m", "big.gguf"),
                cgroup="0::/other.slice")
        self.add_lock("frontdoor", "q0", holder=9100,
                      payload=_payload(9100, "frontdoor", "q0", tag="frontdoor-decode"))
        result = P.claim_witness_preflight(self.scope(), self.sources())
        self.assertEqual(result.verdict, P.FAIL)
        self.assertEqual(result.basis, P.BASIS_CLAIM_WITNESS)
        self.assertEqual(len(result.findings), 1)
        finding = result.findings[0]
        self.assertIn("q0", finding.what)
        self.assertIn("frontdoor", finding.what)
        self.assertIn("9100", finding.whose)
        self.assertIn("llama-server", finding.whose)
        self.assertIn("frontdoor-decode", finding.whose)
        self.assertIn("/other.slice", finding.whose)

    def test_our_own_ancestors_claim_is_not_concurrent_inference(self):
        """The canonical bench runs under `region-lock run -- ...`: the holder of
        our own claim is our PARENT."""
        _mkproc(self.proc_root, 4000, ppid=1, comm="region-lock")
        _write_stat(self.proc_root, self.self_pid, 4000, "python3")
        self.add_lock("bench-canonical", "q0", holder=4000,
                      payload=_payload(4000, "bench-canonical", "q0"))
        result = P.claim_witness_preflight(self.scope(), self.sources())
        self.assertEqual(result.verdict, P.PASS)
        self.assertEqual(result.findings, ())

    def test_stale_payload_without_a_live_flock_is_debris_not_a_claim(self):
        """A holder killed with SIGKILL leaves its JSON behind. The flock is the
        fact; treating debris as a claim would block every future run forever."""
        self.add_lock("worker_general", "q1", holder=None,
                      payload=_payload(9999, "worker_general", "q1"))
        result = P.claim_witness_preflight(self.scope(), self.sources())
        self.assertEqual(result.verdict, P.PASS)
        self.assertTrue(any("stale debris" in n for n in result.notes))

    def test_blocked_waiter_is_not_counted_as_a_holder(self):
        """`N: -> FLOCK ...` shifts every field by one; misparsing it invents a
        FAIL out of a process that is queued, not running."""
        self.add_lock("bench", "q2", holder=None, blocked_waiter=9200, payload="")
        claims = P.read_region_claims(self.lock_dir, self.proc())
        claim = next(c for c in claims if c.region == "q2")
        self.assertFalse(claim.held)
        self.assertEqual(claim.holders.waiter_pids, (9200,))
        result = P.claim_witness_preflight(self.scope(), self.sources())
        self.assertEqual(result.verdict, P.PASS)

    def test_claim_outside_the_requested_scope_is_ignored(self):
        _mkproc(self.proc_root, 9300, ppid=1, comm="llama-server")
        self.add_lock("worker_general", "q3", holder=9300,
                      payload=_payload(9300, "worker_general", "q3"))
        narrow = P.PreflightScope.cpu("quarter-bench", ["q0"])
        self.assertEqual(P.claim_witness_preflight(narrow, self.sources()).verdict, P.PASS)
        wide = P.PreflightScope.whole_machine_cpu("full-machine-bench")
        self.assertEqual(P.claim_witness_preflight(wide, self.sources()).verdict, P.FAIL)

    def test_global_cross_role_mutex_is_reported_as_such(self):
        _mkproc(self.proc_root, 9400, ppid=1, comm="llama-server")
        self.add_lock("GLOBAL", "q0", holder=9400)
        result = P.claim_witness_preflight(self.scope(), self.sources())
        self.assertEqual(result.verdict, P.FAIL)
        self.assertIn("GLOBAL mutex", result.findings[0].what)

    def test_unattributed_ofd_holder_fails_and_says_so(self):
        """An OFD lock reports pid -1: held, but not attributable. Guessing a
        holder would be worse than saying we cannot name one."""
        self.add_lock("bench", "q0", holder=-1, klass="OFDLCK")
        result = P.claim_witness_preflight(self.scope(), self.sources())
        self.assertEqual(result.verdict, P.FAIL)
        self.assertIn("UNATTRIBUTED", result.findings[0].whose)

    def test_unknown_payload_schema_version_degrades_attribution_not_occupancy(self):
        _mkproc(self.proc_root, 9500, ppid=1, comm="llama-server")
        self.add_lock("frontdoor", "q0", holder=9500,
                      payload=json.dumps({"schema_version": 99, "pid": 9500}))
        result = P.claim_witness_preflight(self.scope(), self.sources())
        self.assertEqual(result.verdict, P.FAIL)
        self.assertTrue(any("schema_version" in n for n in result.notes))

    def test_corrupt_payload_is_noted_and_the_flock_still_counts(self):
        _mkproc(self.proc_root, 9600, ppid=1, comm="llama-server")
        self.add_lock("frontdoor", "q0", holder=9600, payload="{not json")
        result = P.claim_witness_preflight(self.scope(), self.sources())
        self.assertEqual(result.verdict, P.FAIL)
        self.assertTrue(any("not valid JSON" in n for n in result.notes))

    def test_payload_keys_match_the_mirrored_on_disk_contract(self):
        """Pins the shape this module mirrors from cpu_region_lock, so a drift
        shows up here rather than as a silently unattributed FAIL."""
        loaded = json.loads(_payload(1, "r", "q0"))
        self.assertEqual(tuple(sorted(loaded)), tuple(sorted(_PAYLOAD_KEYS)))


# ===========================================================================
# 6. Claim witness — COULD_NOT_CHECK paths
# ===========================================================================

class ClaimRootUnreadableTest(_Fixture):

    def test_missing_claim_root_is_could_not_check(self):
        sources = self.sources(region_lock_dir=self.root / "no-such-dir")
        result = P.claim_witness_preflight(self.scope(), sources)
        self.assertEqual(result.verdict, P.COULD_NOT_CHECK)
        self.assertFalse(result.passed)
        self.assertTrue(any("does not exist" in r for r in result.reasons))

    def test_claim_root_that_is_a_file_is_could_not_check(self):
        path = self.root / "not-a-dir"
        path.write_text("")
        result = P.claim_witness_preflight(self.scope(), self.sources(region_lock_dir=path))
        self.assertEqual(result.verdict, P.COULD_NOT_CHECK)

    @unittest.skipIf(os.geteuid() == 0, "root can read a 0o000 directory")
    def test_permission_denied_claim_root_is_could_not_check(self):
        denied = self.root / "denied"
        denied.mkdir()
        (denied / "cpu_region.frontdoor.q0.lock").write_text("")
        os.chmod(denied, 0o000)
        self.addCleanup(os.chmod, denied, 0o755)
        result = P.claim_witness_preflight(self.scope(), self.sources(region_lock_dir=denied))
        self.assertEqual(result.verdict, P.COULD_NOT_CHECK)
        self.assertFalse(result.passed)

    def test_empty_namespace_is_could_not_check_not_no_claims(self):
        """An empty region-lock directory on this host means the path is wrong
        far more often than it means the fleet is idle. Reading it as 'no
        claims' would manufacture a PASS out of a misconfiguration."""
        result = P.claim_witness_preflight(self.scope(), self.sources())
        self.assertEqual(result.verdict, P.COULD_NOT_CHECK)
        self.assertTrue(any("empty namespace" in r for r in result.reasons))

    def test_empty_namespace_can_be_accepted_only_deliberately(self):
        sources = self.sources(require_nonempty_namespace=False)
        result = P.claim_witness_preflight(self.scope(), sources)
        self.assertEqual(result.verdict, P.PASS)

    def test_missing_proc_locks_is_could_not_check(self):
        self.add_lock("frontdoor", "q0", holder=None, payload="")
        (self.proc_root / "locks").unlink()
        result = P.claim_witness_preflight(self.scope(), self.sources())
        self.assertEqual(result.verdict, P.COULD_NOT_CHECK)

    def test_read_region_claims_raises_rather_than_returning_empty(self):
        with self.assertRaises(P.PreflightUnavailable):
            P.read_region_claims(self.root / "nope", self.proc())
        with self.assertRaises(P.PreflightUnavailable):
            P.read_region_claims(self.lock_dir, self.proc())


# ===========================================================================
# 7. Claim witness — GPU device claim (the substrate that does not exist yet)
# ===========================================================================

class GpuClaimTest(_Fixture):

    def test_gpu_scope_without_a_reader_is_could_not_check_never_pass(self):
        """§2.5: no cross-process GPU device claim exists. Its silence means
        nothing, so reporting an unclaimed GPU as free would fabricate a P-GPU-1
        precondition."""
        scope = P.PreflightScope.gpu("gpu-bench", ["gfx90a:0"])
        result = P.claim_witness_preflight(scope, self.sources())
        self.assertEqual(result.verdict, P.COULD_NOT_CHECK)
        self.assertFalse(result.passed)
        self.assertTrue(any("device-claim reader" in r for r in result.reasons))

    def test_foreign_gpu_claim_fails(self):
        witness = P.GpuClaimWitness(device_id="gfx90a:0", holder_pid=8800,
                                    holder_label="autopilot-eval",
                                    source="gpu_device_claim.v1")
        scope = P.PreflightScope.gpu("gpu-bench", ["gfx90a:0"])
        sources = self.sources(gpu_claim_reader=lambda: [witness])
        result = P.claim_witness_preflight(scope, sources)
        self.assertEqual(result.verdict, P.FAIL)
        self.assertIn("gfx90a:0", result.findings[0].what)
        self.assertIn("autopilot-eval", result.findings[0].whose)

    def test_our_own_gpu_claim_passes(self):
        witness = P.GpuClaimWitness(device_id="gfx90a:0", holder_pid=self.self_pid,
                                    holder_label="this-evaluator",
                                    source="gpu_device_claim.v1")
        scope = P.PreflightScope.gpu("gpu-bench", ["gfx90a:0"])
        sources = self.sources(gpu_claim_reader=lambda: [witness])
        self.assertEqual(P.claim_witness_preflight(scope, sources).verdict, P.PASS)

    def test_claim_on_a_device_outside_scope_is_ignored(self):
        witness = P.GpuClaimWitness(device_id="gfx90a:1", holder_pid=8800,
                                    holder_label="other", source="gpu_device_claim.v1")
        scope = P.PreflightScope.gpu("gpu-bench", ["gfx90a:0"])
        sources = self.sources(gpu_claim_reader=lambda: [witness])
        self.assertEqual(P.claim_witness_preflight(scope, sources).verdict, P.PASS)

    def test_reader_that_raises_is_could_not_check_not_pass(self):
        def broken():
            raise P.PreflightUnavailable("device claim socket down")

        scope = P.PreflightScope.gpu("gpu-bench", ["gfx90a:0"])
        result = P.claim_witness_preflight(scope, self.sources(gpu_claim_reader=broken))
        self.assertEqual(result.verdict, P.COULD_NOT_CHECK)

    def test_reader_that_raises_an_unexpected_error_is_still_could_not_check(self):
        def broken():
            raise KeyError("holder")

        scope = P.PreflightScope.gpu("gpu-bench", ["gfx90a:0"])
        result = P.claim_witness_preflight(scope, self.sources(gpu_claim_reader=broken))
        self.assertEqual(result.verdict, P.COULD_NOT_CHECK)
        self.assertTrue(any("KeyError" in r for r in result.reasons))

    def test_cpu_pass_plus_gpu_blind_spot_is_could_not_check_overall(self):
        self.add_lock("frontdoor", "q0", holder=None, payload="")
        scope = P.PreflightScope(label="mixed", cpu_regions=None,
                                 gpu_devices=frozenset({"gfx90a:0"}))
        result = P.claim_witness_preflight(scope, self.sources())
        self.assertEqual(result.verdict, P.COULD_NOT_CHECK)


# ===========================================================================
# 8. Layer preference — claim witness beats the scan
# ===========================================================================

def _exploding_scanner(*args, **kwargs):
    raise AssertionError("the interim name-pattern scanner must not run when "
                         "claim witness could evaluate")


class LayerPreferenceTest(_Fixture):

    def test_claim_pass_does_not_consult_the_scan(self):
        self.add_lock("frontdoor", "q0", holder=None, payload="")
        result = P.preflight(self.scope(), self.sources(),
                             interim_scan=P.InterimScan.ALLOW_LABELLED,
                             scanner=_exploding_scanner)
        self.assertEqual(result.verdict, P.PASS)
        self.assertEqual(result.basis, P.BASIS_CLAIM_WITNESS)
        self.assertIsNone(result.scan)

    def test_claim_fail_does_not_consult_the_scan(self):
        _mkproc(self.proc_root, 9700, ppid=1, comm="llama-server")
        self.add_lock("frontdoor", "q0", holder=9700,
                      payload=_payload(9700, "frontdoor", "q0"))
        result = P.preflight(self.scope(), self.sources(),
                             interim_scan=P.InterimScan.ALLOW_LABELLED,
                             scanner=_exploding_scanner)
        self.assertEqual(result.verdict, P.FAIL)
        self.assertEqual(result.basis, P.BASIS_CLAIM_WITNESS)

    def test_interim_is_denied_by_default(self):
        """Sliding from claim witness to a name-pattern scan without the call
        site saying so is the silent-degradation pattern; the default refuses."""
        sources = self.sources(region_lock_dir=self.root / "gone")
        result = P.preflight(self.scope(), sources, scanner=_exploding_scanner)
        self.assertEqual(result.verdict, P.COULD_NOT_CHECK)
        self.assertEqual(result.basis, P.BASIS_CLAIM_WITNESS)
        self.assertTrue(any("was NOT consulted" in n for n in result.notes))

    def test_interim_runs_only_when_claim_witness_could_not_check(self):
        _mkproc(self.proc_root, 9800, ppid=1, comm="llama-server",
                argv=("/opt/llama/llama-server", "-m", "x.gguf"))
        sources = self.sources(region_lock_dir=self.root / "gone")
        result = P.preflight(self.scope(), sources,
                             interim_scan=P.InterimScan.ALLOW_LABELLED)
        self.assertEqual(result.verdict, P.FAIL)
        self.assertEqual(result.basis, P.BASIS_INTERIM_PROCESS_SCAN)
        self.assertIsNotNone(result.scan)
        self.assertTrue(any("does not exist" in r for r in result.reasons),
                        msg="the fallback must carry why the preferred layer failed")

    def test_interim_pass_is_labelled_as_the_weaker_instrument(self):
        sources = self.sources(region_lock_dir=self.root / "gone")
        result = P.preflight(self.scope(), sources,
                             interim_scan=P.InterimScan.ALLOW_LABELLED)
        self.assertEqual(result.verdict, P.PASS)
        self.assertEqual(result.basis, P.BASIS_INTERIM_PROCESS_SCAN)
        self.assertTrue(any("INTERIM" in n for n in result.notes))

    def test_interim_scan_flag_must_be_the_enum(self):
        with self.assertRaises(TypeError):
            P.preflight(self.scope(), self.sources(), interim_scan=True)

    def test_require_no_concurrent_inference_raises_on_indeterminate(self):
        sources = self.sources(region_lock_dir=self.root / "gone")
        with self.assertRaises(P.PreflightIndeterminate):
            P.require_no_concurrent_inference(self.scope(), sources)

    def test_require_no_concurrent_inference_returns_the_attestation_on_pass(self):
        self.add_lock("frontdoor", "q0", holder=None, payload="")
        result = P.require_no_concurrent_inference(self.scope(), self.sources())
        self.assertEqual(result.verdict, P.PASS)


# ===========================================================================
# 9. The interim enumerator itself
# ===========================================================================

class InterimScanTest(_Fixture):

    def test_matches_by_executable_basename(self):
        _mkproc(self.proc_root, 9900, ppid=1, comm="llama-server",
                argv=("/mnt/raid0/llm/llama.cpp/build/bin/llama-server", "-m", "big.gguf"))
        scan = P.interim_process_scan(proc=self.proc(), owned=P.read_own_scope(self.proc()))
        self.assertEqual([o.pid for o in scan.inference_like()], [9900])

    def test_our_own_processes_are_owned_not_concurrent(self):
        _mkproc(self.proc_root, 9910, ppid=self.self_pid, comm="llama-bench",
                argv=("/opt/llama/llama-bench", "-m", "x.gguf"))
        owned = P.read_own_scope(self.proc())
        scan = P.interim_process_scan(proc=self.proc(), owned=owned)
        self.assertEqual(scan.inference_like(), ())
        self.assertEqual(scan.observations[0].classification, P.Classification.OWNED)

    def test_earlyoom_guard_is_not_inference_in_exe_mode(self):
        """INC-20260731: a pattern sweep killed earlyoom because its own argv
        contains `--ignore ^(llama-server|sd-server)$`."""
        _mkproc(self.proc_root, 9920, ppid=1, comm="earlyoom",
                argv=("/usr/bin/earlyoom", "--ignore", "^(llama-server|sd-server)$"))
        scan = P.interim_process_scan(proc=self.proc())
        self.assertEqual(scan.observations, ())

    def test_earlyoom_guard_is_classified_not_counted_in_full_cmdline_mode(self):
        _mkproc(self.proc_root, 9930, ppid=1, comm="earlyoom",
                argv=("/usr/bin/earlyoom", "--ignore", "^(llama-server|sd-server)$"))
        scan = P.interim_process_scan(proc=self.proc(),
                                      match_field=P.MatchField.FULL_CMDLINE)
        self.assertEqual(scan.inference_like(), ())
        self.assertEqual(scan.observations[0].classification, P.Classification.GUARD_ARGV_ONLY)

    def test_generic_ignore_flag_value_is_a_guard_mention(self):
        _mkproc(self.proc_root, 9940, ppid=1, comm="watchdog",
                argv=("/usr/bin/watchdog", "--exclude", "llama-server"))
        scan = P.interim_process_scan(proc=self.proc(),
                                      match_field=P.MatchField.FULL_CMDLINE)
        self.assertEqual(scan.inference_like(), ())
        self.assertEqual(scan.observations[0].classification, P.Classification.GUARD_ARGV_ONLY)

    def test_an_argument_mention_is_not_an_inference_process(self):
        _mkproc(self.proc_root, 9950, ppid=1, comm="tail",
                argv=("/usr/bin/tail", "-f", "/var/log/llama-server.log"))
        scan = P.interim_process_scan(proc=self.proc(),
                                      match_field=P.MatchField.FULL_CMDLINE)
        self.assertEqual(scan.inference_like(), ())
        self.assertEqual(scan.observations[0].classification, P.Classification.ARGV_MENTION_ONLY)

    def test_zombie_with_no_cmdline_is_matched_on_comm(self):
        """`bench-cpu.md`'s precondition is explicitly a ZOMBIE check, and a
        zombie has no readable cmdline."""
        pid_dir = self.proc_root / "9960"
        pid_dir.mkdir()
        _write_stat(self.proc_root, 9960, 1, "llama-server")
        (pid_dir / "cmdline").write_bytes(b"")
        (pid_dir / "cgroup").write_text("0::/other.slice\n")
        scan = P.interim_process_scan(proc=self.proc())
        self.assertEqual([o.pid for o in scan.inference_like()], [9960])

    def test_vanished_pid_is_recorded_not_fatal(self):
        (self.proc_root / "9970").mkdir()  # a pid dir with no stat: gone mid-scan
        scan = P.interim_process_scan(proc=self.proc())
        self.assertIn(9970, scan.vanished_pids)
        self.assertEqual(scan.inference_like(), ())

    @unittest.skipIf(os.geteuid() == 0, "root can read a 0o000 file")
    def test_unreadable_process_downgrades_the_scan_to_could_not_check(self):
        _mkproc(self.proc_root, 9980, ppid=1, comm="python3")
        os.chmod(self.proc_root / "9980" / "stat", 0o000)
        self.addCleanup(os.chmod, self.proc_root / "9980" / "stat", 0o644)
        sources = self.sources(region_lock_dir=self.root / "gone")
        result = P.preflight(self.scope(), sources,
                             interim_scan=P.InterimScan.ALLOW_LABELLED)
        self.assertEqual(result.verdict, P.COULD_NOT_CHECK)
        self.assertIn(9980, result.scan.unreadable_pids)

    def test_empty_pattern_set_is_refused(self):
        with self.assertRaises(ValueError):
            P.interim_process_scan([], proc=self.proc())

    def test_blank_pattern_is_refused(self):
        with self.assertRaises(ValueError):
            P.interim_process_scan(["llama-server", ""], proc=self.proc())

    def test_default_patterns_cover_the_protocol_named_binaries(self):
        for name in ("llama-server", "llama-bench", "llama-cli"):
            self.assertIn(name, P.DEFAULT_INFERENCE_EXE_PATTERNS)

    def test_missing_proc_root_raises(self):
        proc = P.ProcSource(root=self.root / "nope", self_pid=self.self_pid)
        with self.assertRaises(P.PreflightUnavailable):
            P.interim_process_scan(proc=proc)


# ===========================================================================
# 10. The result is durable evidence
# ===========================================================================

class AttestationTest(_Fixture):

    def test_result_dict_is_canonical_json_safe(self):
        """The result is journaled verbatim as the precondition attestation, so
        it must survive `schemas.canonical_json` — which refuses tuples,
        non-string keys and non-finite floats."""
        _mkproc(self.proc_root, 9990, ppid=1, comm="llama-server",
                argv=("/opt/llama/llama-server", "-m", "x.gguf"))
        self.add_lock("frontdoor", "q0", holder=9990,
                      payload=_payload(9990, "frontdoor", "q0"))
        result = P.claim_witness_preflight(self.scope(), self.sources())
        encoded = S.canonical_json(result.to_dict())
        self.assertEqual(json.loads(encoded)["verdict"], P.FAIL)
        self.assertEqual(json.loads(encoded)["basis"], P.BASIS_CLAIM_WITNESS)

    def test_scan_result_dict_is_canonical_json_safe(self):
        _mkproc(self.proc_root, 9991, ppid=1, comm="llama-server",
                argv=("/opt/llama/llama-server",))
        sources = self.sources(region_lock_dir=self.root / "gone")
        result = P.preflight(self.scope(), sources,
                             interim_scan=P.InterimScan.ALLOW_LABELLED)
        S.canonical_json(result.to_dict())

    def test_attestation_records_which_layer_decided(self):
        self.add_lock("frontdoor", "q0", holder=None, payload="")
        claim_result = P.preflight(self.scope(), self.sources())
        self.assertEqual(claim_result.to_dict()["basis"], P.BASIS_CLAIM_WITNESS)
        scan_result = P.preflight(self.scope(),
                                  self.sources(region_lock_dir=self.root / "gone"),
                                  interim_scan=P.InterimScan.ALLOW_LABELLED)
        self.assertEqual(scan_result.to_dict()["basis"], P.BASIS_INTERIM_PROCESS_SCAN)

    def test_observed_at_is_an_aware_utc_stamp(self):
        self.add_lock("frontdoor", "q0", holder=None, payload="")
        result = P.claim_witness_preflight(self.scope(), self.sources())
        self.assertTrue(result.observed_at.endswith("Z"), msg=result.observed_at)

    def test_clock_is_injectable_for_deterministic_replay(self):
        self.add_lock("frontdoor", "q0", holder=None, payload="")
        result = P.claim_witness_preflight(self.scope(), self.sources(),
                                           now=lambda: "2026-08-03T12:00:00Z")
        self.assertEqual(result.observed_at, "2026-08-03T12:00:00Z")


# ===========================================================================
# 11. Environment resolution
# ===========================================================================

class LockDirResolutionTest(unittest.TestCase):

    def test_env_override_precedence_matches_the_orchestrator(self):
        self.assertEqual(
            P.default_region_lock_dir({"ORCHESTRATOR_TMP_DIR": "/a",
                                       "ORCHESTRATOR_PATHS_TMP_DIR": "/b"}),
            Path("/a"))
        self.assertEqual(
            P.default_region_lock_dir({"ORCHESTRATOR_PATHS_TMP_DIR": "/b"}),
            Path("/b"))
        self.assertEqual(P.default_region_lock_dir({}), Path("/mnt/raid0/llm/tmp"))

    def test_from_environment_builds_sources_without_touching_the_host(self):
        sources = P.ClaimSources.from_environment({"ORCHESTRATOR_TMP_DIR": "/x"})
        self.assertEqual(sources.region_lock_dir, Path("/x"))
        self.assertIsNone(sources.gpu_claim_reader)


def _nobody_is_owned() -> "P.OwnedScope":
    """An OwnedScope that owns nothing, so a real live holder reads as foreign.

    Injected rather than derived: giving `ProcSource` a fake `self_pid` to make
    ourselves look foreign does not work — `self_pid=1` makes every process in
    the namespace a descendant, i.e. owned.
    """
    return P.OwnedScope(self_pid=os.getpid(), cgroup=None, pids=frozenset(),
                        reasons={}, incomplete=())


# ===========================================================================
# 12. Adversarial red-team regressions (2026-08-03)
#
# Every test below was written against a CONFIRMED defect: each one failed on
# the module as first written. They are grouped by the axis that found them.
# ===========================================================================

class PatternOrderFailOpenTest(_Fixture):
    """A live inference process must not be excused by an EARLIER pattern.

    Defect: `interim_process_scan` stopped at the FIRST pattern that produced
    any classification, so the ORDER of `DEFAULT_INFERENCE_EXE_PATTERNS` decided
    whether a process counted. `sd-server --lora-dir /models/llama-cli/loras`
    matched "llama-cli" (index 2) as ARGV_MENTION_ONLY and broke out before
    "sd-server" (index 7) could match argv[0] — a running inference server read
    as PASS.
    """

    def test_argv0_match_beats_an_earlier_patterns_mere_mention(self):
        _mkproc(self.proc_root, 9991, ppid=1, comm="sd-server",
                argv=("/opt/bin/sd-server", "--lora-dir", "/models/llama-cli/loras"))
        scan = P.interim_process_scan(proc=self.proc(),
                                      match_field=P.MatchField.FULL_CMDLINE)
        self.assertEqual([o.pid for o in scan.inference_like()], [9991])
        self.assertEqual(scan.observations[0].matched_pattern, "sd-server")

    def test_the_scan_verdict_is_fail_not_pass(self):
        _mkproc(self.proc_root, 9992, ppid=1, comm="whisper-server",
                argv=("/opt/bin/whisper-server", "--model", "/m/llama-server-notes.bin"))
        result = P.interim_scan_preflight(self.scope(), proc=self.proc(),
                                          match_field=P.MatchField.FULL_CMDLINE)
        self.assertEqual(result.verdict, P.FAIL)

    def test_guard_classification_still_wins_over_a_bare_mention(self):
        """The INC-20260731 defence must survive the strongest-match rule."""
        _mkproc(self.proc_root, 9993, ppid=1, comm="earlyoom",
                argv=("/usr/bin/earlyoom", "--ignore", "^(llama-server|sd-server)$"))
        scan = P.interim_process_scan(proc=self.proc(),
                                      match_field=P.MatchField.FULL_CMDLINE)
        self.assertEqual(scan.inference_like(), ())
        self.assertEqual(scan.observations[0].classification, P.Classification.GUARD_ARGV_ONLY)


class GpuScopeCannotBeScannedTest(_Fixture):
    """The interim scan must never PASS a scope it structurally cannot see.

    Defect: with `InterimScan.ALLOW_LABELLED`, a GPU-scoped preflight with no
    device-claim reader fell through to a /proc name scan which opens no device,
    reads no claim, and knows nothing about the GPU — and returned PASS. That is
    exactly the fabricated P-GPU-1 precondition the module says it never makes.
    """

    def test_gpu_scope_with_no_reader_is_could_not_check_even_when_scanning(self):
        gpu_scope = P.PreflightScope.gpu("gpu-bench", ["0000:c1:00.0"])
        result = P.preflight(gpu_scope, self.sources(),
                             interim_scan=P.InterimScan.ALLOW_LABELLED)
        self.assertEqual(result.verdict, P.COULD_NOT_CHECK)
        self.assertFalse(result.passed)
        self.assertTrue(any("cannot witness" in r for r in result.reasons),
                        msg=f"reasons: {result.reasons}")

    def test_a_scanned_gpu_scope_still_fails_when_inference_is_seen(self):
        """Downgrading PASS must not also downgrade FAIL."""
        _mkproc(self.proc_root, 9994, ppid=1, comm="llama-server",
                argv=("/opt/bin/llama-server", "-m", "x.gguf"))
        gpu_scope = P.PreflightScope.gpu("gpu-bench", ["0000:c1:00.0"])
        result = P.preflight(gpu_scope, self.sources(),
                             interim_scan=P.InterimScan.ALLOW_LABELLED)
        self.assertEqual(result.verdict, P.FAIL)


class AuditEscapeHatchTest(unittest.TestCase):
    """`getattr` was banned; its four synonyms were not.

    Defect: the module claimed "no dynamic escape hatch", but
    `vars(os)['kill']`, `os.__dict__['kill']`, `importlib.import_module(...)`,
    `operator.attrgetter('kill')` and `runpy` all audited PASS.
    """

    def _audit(self, body: str):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        path = Path(tmp.name) / "candidate.py"
        path.write_text(body, encoding="utf-8")
        return P.audit_no_signalling_capability(path)

    def test_vars_indirection_is_caught(self):
        self.assertEqual(self._audit("import os\nvars(os)['ki' 'll'](1, 9)\n").outcome, P.FAIL)

    def test_dunder_dict_indirection_is_caught(self):
        self.assertEqual(self._audit("import os\nos.__dict__['x'](1, 9)\n").outcome, P.FAIL)

    def test_globals_indirection_is_caught(self):
        self.assertEqual(self._audit("def f():\n    globals()['g'](1)\n").outcome, P.FAIL)

    def test_importlib_indirection_is_caught(self):
        self.assertEqual(self._audit("import importlib\n").outcome, P.FAIL)

    def test_operator_attrgetter_indirection_is_caught(self):
        self.assertEqual(self._audit("import operator\n").outcome, P.FAIL)

    def test_runpy_is_caught(self):
        self.assertEqual(self._audit("import runpy\nrunpy.run_module('x')\n").outcome, P.FAIL)

    def test_the_module_itself_still_audits_clean_under_the_wider_net(self):
        check = P.audit_no_signalling_capability()
        self.assertEqual(check.outcome, P.PASS, msg=f"violations: {check.reasons}")


class AttestationDurabilityTest(_Fixture):
    """An attestation that cannot be written is an outcome that did not happen.

    Defect 1: `json.loads` accepts the non-standard `NaN`/`Infinity` literals
    and the payload was embedded verbatim, so one poisoned lock file made
    `schemas.canonical_json(result.to_dict())` RAISE — the FAIL was correct and
    unrecordable (§4 invariant 7).

    Defect 2: `require_no_concurrent_inference` discarded the result on FAIL and
    COULD_NOT_CHECK — the only two outcomes invariant 7 exists for.
    """

    def test_non_finite_payload_keeps_the_attestation_canonicalisable(self):
        self.add_lock("frontdoor", "q0", holder=9600,
                      payload='{"schema_version": 1, "pid": 9600, "started_at": NaN}')
        result = P.claim_witness_preflight(self.scope(), self.sources())
        self.assertEqual(result.verdict, P.FAIL)
        S.canonical_json(result.to_dict())          # must not raise
        self.assertTrue(any("non-finite" in n for n in result.notes),
                        msg=f"notes: {result.notes}")

    def test_infinity_payload_is_handled_too(self):
        self.add_lock("worker", "q1", holder=9601,
                      payload='{"schema_version": 1, "regions": [Infinity]}')
        result = P.claim_witness_preflight(self.scope(), self.sources())
        S.canonical_json(result.to_dict())

    def test_the_flock_still_counts_when_the_payload_is_dropped(self):
        """Occupancy is the flock; attribution degrading must not free the region."""
        self.add_lock("frontdoor", "q0", holder=9602, payload='{"schema_version": 1, "x": NaN}')
        result = P.claim_witness_preflight(self.scope(), self.sources())
        self.assertEqual(result.verdict, P.FAIL)
        self.assertTrue(result.region_claims[0].held)

    def test_fail_attestation_rides_on_the_exception(self):
        self.add_lock("frontdoor", "q0", holder=9603, payload=_payload(9603, "frontdoor", "q0"))
        with self.assertRaises(P.ConcurrentInferenceDetected) as caught:
            P.require_no_concurrent_inference(self.scope(), self.sources())
        self.assertIsNotNone(caught.exception.result)
        self.assertEqual(caught.exception.result.verdict, P.FAIL)
        S.canonical_json(caught.exception.result.to_dict())

    def test_could_not_check_attestation_rides_on_the_exception(self):
        sources = self.sources(region_lock_dir=self.root / "absent")
        with self.assertRaises(P.PreflightIndeterminate) as caught:
            P.require_no_concurrent_inference(self.scope(), sources)
        self.assertIsNotNone(caught.exception.result)
        self.assertEqual(caught.exception.result.verdict, P.COULD_NOT_CHECK)
        S.canonical_json(caught.exception.result.to_dict())


class NamespaceDriftTest(_Fixture):
    """A namespace we cannot PARSE is not an empty namespace.

    Defect: `require_nonempty_namespace` only refused a namespace with zero
    matching files. Files that matched the glob but not the
    `cpu_region.<role>.<region>.lock` shape were skipped silently, so a shape
    drift produced `claims == []` — a PASS manufactured out of a rename.
    """

    def test_all_names_unparseable_is_could_not_check_not_pass(self):
        for index in range(3):
            (self.lock_dir / f"cpu_region..q{index}.lock").write_text("")
        result = P.claim_witness_preflight(self.scope(), self.sources())
        self.assertEqual(result.verdict, P.COULD_NOT_CHECK)
        self.assertFalse(result.passed)
        self.assertTrue(any("naming contract" in r for r in result.reasons),
                        msg=f"reasons: {result.reasons}")

    def test_one_parseable_file_is_enough_to_evaluate(self):
        (self.lock_dir / "cpu_region..q0.lock").write_text("")
        self.add_lock("frontdoor", "q0", holder=None, payload="")
        result = P.claim_witness_preflight(self.scope(), self.sources())
        self.assertEqual(result.verdict, P.PASS)


class HolderAttributionTest(_Fixture):
    """The flock is the fact; a payload that names a different pid is not it."""

    def test_payload_pid_disagreeing_with_the_live_holder_is_noted(self):
        self.add_lock("frontdoor", "q0", holder=9700,
                      payload=_payload(4242, "frontdoor", "q0", tag="stale-tag"))
        result = P.claim_witness_preflight(self.scope(), self.sources())
        self.assertEqual(result.verdict, P.FAIL)
        self.assertTrue(any("not trustworthy" in n for n in result.notes),
                        msg=f"notes: {result.notes}")

    def test_matching_payload_pid_produces_no_such_note(self):
        self.add_lock("frontdoor", "q0", holder=9700,
                      payload=_payload(9700, "frontdoor", "q0"))
        result = P.claim_witness_preflight(self.scope(), self.sources())
        self.assertFalse(any("not trustworthy" in n for n in result.notes))


class ProcSourceDefaultTest(unittest.TestCase):
    """`ClaimSources.proc` must not cache a pid taken at IMPORT time.

    Defect: the default was a bare `ProcSource()` evaluated once at class
    creation, so every `ClaimSources` shared one instance whose `self_pid` was
    the importing process's. After a fork in the host program the default names
    the PARENT, whose owned set is a SUPERSET of the child's — over-broad
    ownership, i.e. a foreign claim read as ours: a false PASS. Only the
    opposite error (a false FAIL) is supposed to be reachable.
    """

    def test_default_proc_source_is_per_instance(self):
        first = P.ClaimSources(region_lock_dir=Path("/x"))
        second = P.ClaimSources(region_lock_dir=Path("/x"))
        self.assertIsNot(first.proc, second.proc)

    def test_default_proc_source_reports_the_live_pid(self):
        self.assertEqual(P.ClaimSources(region_lock_dir=Path("/x")).proc.self_pid, os.getpid())


class RunningAsPidOneTest(_Fixture):
    """Being pid 1 must not make the whole machine "ours".

    Defect: `read_own_scope` excluded pid 1 as an ANCESTOR ("it is everyone's
    ancestor") but not as SELF. A preflight running as pid 1 of a PID namespace
    — a container entrypoint — walks its descendants and reaches every process
    in the namespace, so every claim is owned and the verdict is a guaranteed
    PASS. Ownership is genuinely undecidable from there, so it must be
    COULD_NOT_CHECK, not PASS.
    """

    def test_read_own_scope_refuses_to_run_as_pid_1(self):
        with self.assertRaises(P.PreflightUnavailable):
            P.read_own_scope(P.ProcSource(root=self.proc_root, self_pid=1))

    def test_a_foreign_claim_is_not_swallowed_by_pid_1_ownership(self):
        _mkproc(self.proc_root, 9800, ppid=1, comm="llama-server",
                argv=("/opt/bin/llama-server", "-m", "x.gguf"))
        self.add_lock("frontdoor", "q0", holder=9800,
                      payload=_payload(9800, "frontdoor", "q0"))
        sources = self.sources(proc=P.ProcSource(root=self.proc_root, self_pid=1))
        result = P.claim_witness_preflight(self.scope(), sources)
        self.assertEqual(result.verdict, P.COULD_NOT_CHECK)
        self.assertFalse(result.passed)


class RealKernelLockEncodingTest(unittest.TestCase):
    """END-TO-END against the real kernel, not against our own fixture.

    Gap this closes: every other lock test feeds `/proc/locks` lines built by
    `_locks_line`, which formats the device with the SAME `"%02x:%02x"` the code
    under test uses — so the suite would pass unchanged if that encoding did not
    match what the kernel actually prints, and every claim would read "not
    held" (a permanent, silent PASS). This test takes a REAL `flock` on a REAL
    file in this process and requires the real `/proc/locks` line to be found
    under the key `_lock_key` computes.

    No process is started, stopped or signalled: the lock is taken and released
    on this interpreter's own file descriptor.
    """

    def test_computed_lock_key_finds_a_real_flock_in_real_proc_locks(self):
        import fcntl  # local: the module under test must never import it
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        lock_dir = Path(tmp.name)
        path = lock_dir / "cpu_region.redteam.q0.lock"
        handle = open(path, "a+")                                  # noqa: SIM115
        self.addCleanup(handle.close)
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:                                      # pragma: no cover
            self.skipTest(f"cannot flock on this filesystem: {exc}")
        self.addCleanup(fcntl.flock, handle.fileno(), fcntl.LOCK_UN)

        real_proc = P.ProcSource(root=Path("/proc"), self_pid=os.getpid())
        table = P._read_proc_locks(real_proc)
        key = P._lock_key(path)
        self.assertIn(key, table,
                      msg="computed (dev,inode) key does not match the kernel's /proc/locks "
                          "encoding — every region claim would read as unheld")
        self.assertIn(os.getpid(), table[key].holder_pids)

    def test_a_real_live_flock_is_seen_as_a_foreign_claim(self):
        """The whole pipeline — glob, stat, /proc/locks, ownership — on real
        kernel state. The holder is this process, declared foreign by giving the
        ProcSource a pid that is not ours."""
        import fcntl
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        lock_dir = Path(tmp.name)
        path = lock_dir / "cpu_region.redteam.q0.lock"
        handle = open(path, "a+")                                  # noqa: SIM115
        self.addCleanup(handle.close)
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:                                      # pragma: no cover
            self.skipTest(f"cannot flock on this filesystem: {exc}")
        self.addCleanup(fcntl.flock, handle.fileno(), fcntl.LOCK_UN)

        sources = P.ClaimSources(region_lock_dir=lock_dir,
                                 proc=P.ProcSource(root=Path("/proc"), self_pid=os.getpid()))
        result = P.claim_witness_preflight(P.PreflightScope(label="real-flock"), sources,
                                           owned=_nobody_is_owned())
        self.assertEqual(result.verdict, P.FAIL, msg=f"reasons={result.reasons}")

    @unittest.expectedFailure
    def test_KNOWN_HOLE_unlinking_a_held_lock_file_hides_its_live_holder(self):
        """OPEN DEFECT — this is the project's standing screen, and it succeeds.

        "Can I pass this check by deleting the thing it inspects?" Yes: the
        claim witness enumerates lock FILES and then looks each one up in
        /proc/locks, so unlinking a lock file while its holder is alive flips
        FAIL -> PASS. The flock survives on the unlinked inode and is still
        listed in /proc/locks; only the directory entry the reader depends on is
        gone. `require_nonempty_namespace` does not help — any other lock file
        keeps the namespace non-empty.

        Reaching PASS needs no attacker: a tmp reaper over the lock namespace
        does it. Closing it means reconciling /proc/locks entries on the lock
        dir's device against the visible inodes (and resolving the survivors via
        /proc/<pid>/fd, where "(deleted)" is explicit) — a new subsystem with its
        own permission-dependent COULD_NOT_CHECK paths, so it is left to the
        module owner. This test is the tripwire: when it starts passing
        unexpectedly, the hole is closed and the decorator should be removed.
        """
        import fcntl
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        lock_dir = Path(tmp.name)
        (lock_dir / "cpu_region.decoy.q1.lock").write_text("")
        victim = lock_dir / "cpu_region.redteam.q0.lock"
        handle = open(victim, "a+")                                # noqa: SIM115
        self.addCleanup(handle.close)
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:                                      # pragma: no cover
            self.skipTest(f"cannot flock on this filesystem: {exc}")
        self.addCleanup(fcntl.flock, handle.fileno(), fcntl.LOCK_UN)

        sources = P.ClaimSources(region_lock_dir=lock_dir,
                                 proc=P.ProcSource(root=Path("/proc"), self_pid=os.getpid()))
        scope = P.PreflightScope(label="deleted-lock")
        nobody = _nobody_is_owned()
        self.assertEqual(
            P.claim_witness_preflight(scope, sources, owned=nobody).verdict, P.FAIL)
        victim.unlink()                       # the flock is still held and still in /proc/locks
        self.assertEqual(
            P.claim_witness_preflight(scope, sources, owned=nobody).verdict, P.FAIL,
            msg="a live holder became invisible by deleting the file we inspect")


if __name__ == "__main__":
    unittest.main(verbosity=2)
