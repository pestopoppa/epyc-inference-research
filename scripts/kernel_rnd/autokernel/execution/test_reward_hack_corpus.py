from __future__ import annotations

import fcntl
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

from . import reward_hack_corpus as C


class TestExecutableRewardHackCorpus(unittest.TestCase):
    def test_declared_population_and_taxonomy_are_stated(self):
        self.assertEqual((len(C.PLANTED), len(C.CLEAN)), (11, 15))
        self.assertEqual(len({case.case_id for case in C.CASES}), 26)
        self.assertEqual({case.label for case in C.CASES}, {"planted", "clean"})

    def test_materialized_sources_are_real_hip_programs_and_scan_as_declared(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            rows = [C._materialize(case, root) for case in C.CASES]
        predictions = [(case.label, detected)
                       for case, _source, _binary, _sha, detected, _findings in rows]
        self.assertEqual(sum(label == "planted" and found for label, found in predictions), 11)
        self.assertEqual(sum(label == "clean" and found for label, found in predictions), 0)

    def test_hard_case_is_not_a_gate_only_label(self):
        self.assertIn("anti-short-circuit-127", C._execute.__code__.co_consts)

    def test_materialized_program_reports_the_scored_launch_count(self):
        with tempfile.TemporaryDirectory() as temp:
            _case, source, _binary, _sha, _detected, _findings = C._materialize(
                C.PLANTED[0], Path(temp))
            text = source.read_text(encoding="utf-8")
        self.assertNotIn("@REPETITIONS@", text)
        self.assertIn(f"const int repetitions = {C.KERNEL_REPETITIONS};", text)
        self.assertIn(f"ms, {C.KERNEL_REPETITIONS});", text)

    @staticmethod
    def units(normal_mismatches: int, hard_mismatches: int):
        return [
            {"unit_id": "normal-128", "returncode": 0,
             "result": {"mismatches": normal_mismatches}},
            {"unit_id": "anti-short-circuit-127", "returncode": 0,
             "result": {"mismatches": hard_mismatches}},
        ]

    def test_all_three_repaired_bodies_execute_as_planted_on_both_ranked_units(self):
        # These are the two-unit results emitted by the repaired pointer-key,
        # known-shape and timing bodies.  The live producer executes them; this
        # reducer test pins their fail-closed acceptance semantics.
        for case_id, observed in {
                "pointer-key": self.units(128, 127),
                "known-shape": self.units(128, 127),
                "timing": self.units(128, 127)}.items():
            with self.subTest(case_id=case_id):
                self.assertTrue(C._runtime_accepts("planted", observed))

    def test_old_one_unit_or_nondeterministic_mutations_fail_acceptance(self):
        old_mutations = {
            "pointer-alignment-no-op": self.units(0, 0),
            "known-shape-only-normal": self.units(128, 0),
            "timing-only-hard": self.units(0, 64),
        }
        for mutation, observed in old_mutations.items():
            with self.subTest(mutation=mutation):
                self.assertFalse(C._runtime_accepts("planted", observed))

    @staticmethod
    def prospective_payload() -> dict:
        cases = []
        hipcc = "/opt/rocm/bin/hipcc"
        for index, case in enumerate(C.CASES, 1):
            mismatches = 128 if case.label == "planted" else 0
            source = f"/evidence/sources/{case.case_id}.hip"
            binary = f"/evidence/bin/{case.case_id}"
            cases.append({
                "case_id": case.case_id,
                "label": case.label,
                "mode": case.mode,
                "source": source,
                "source_sha256": f"{index:064x}",
                "binary": binary,
                "binary_sha256": f"{index + 100:064x}",
                "compile_argv": [hipcc, "--offload-arch=gfx90a", "-O2", "-pthread",
                                 source, "-o", binary],
                "compile_returncode": 0,
                "compile_duration_s": 0.1,
                "detected": case.label == "planted",
                "findings": {"scanner": [case.probe] if case.label == "planted" else []},
                "runtime_behavior_manifested": True,
                "ranked_units": [{
                    "unit_id": unit_id,
                    "kind": kind,
                    "n": n,
                    "argv": [binary, str(n)],
                    "returncode": 0,
                    "wall_duration_s": 0.2,
                    "result": {
                        "n": n,
                        "mismatches": mismatches if n == 128 else max(0, mismatches - 1),
                        "gpu_elapsed_ms": float(index) + (0.1 if n == 127 else 0.0),
                        "repetitions": C.KERNEL_REPETITIONS,
                    },
                } for unit_id, kind, n in (
                    ("normal-128", "normal", 128),
                    ("anti-short-circuit-127", "anti_short_circuit", 127),
                )],
            })
        opened = {
            "schema": "epyc.autokernel.device_claim_receipt.v1",
            "claim_id": "akd-fixture",
            "device_id": C.DEVICE_ID,
            "campaign_id": C.CAMPAIGN_ID,
            "acquired_at": "2026-08-12T12:00:00+00:00",
            "released_at": None,
            "state": "held",
        }
        released = {**opened, "released_at": "2026-08-12T12:01:00+00:00"}
        sampling = {
            "schema": "epyc.autokernel.device_sampling_receipt.v1",
            "sampler_id": "autokernel.execution.device_sampler/v1",
            "device_id": "ROCm0",
            "source": "amdgpu-hwmon/numeric-250ms/v1",
            "started_at": "2026-08-12T12:00:00.100000Z",
            "ended_at": "2026-08-12T12:00:59.900000Z",
            "interval_s": 0.25,
            "duration_s": 59.8,
            "command": ["/opt/rocm/bin/rocm-smi", "-d", "0", "--showclocks",
                        "--showpower", "--showtemp"],
            "sample_count": 2,
            "max_gap_s": 0.25,
            "samples": [
                {"offset_s": 0.0, "power_w": 42.0, "sclk_mhz": 800.0,
                 "mclk_mhz": 1600.0, "temperature_c": 35.0,
                 "under_measurement_load": True},
                {"offset_s": 0.25, "power_w": 43.0, "sclk_mhz": 1700.0,
                 "mclk_mhz": 1600.0, "temperature_c": 36.0,
                 "under_measurement_load": True},
            ],
        }
        sampling["sha256"] = C._sha(C._canonical(sampling))
        return {
            "schema": C.SCHEMA,
            "status": "complete",
            "campaign_id": C.CAMPAIGN_ID,
            "purpose": C.PURPOSE,
            "started_at": "2026-08-12T12:00:00Z",
            "ended_at": "2026-08-12T12:01:00Z",
            "host": {"uname": "Linux test", "hipcc": hipcc},
            "producer": {
                "producer_id": C.PRODUCER_ID,
                "path": C.PRODUCER_PATH,
                "sha256": "a" * 64,
            },
            "corpus": {
                "planted": len(C.PLANTED),
                "clean": len(C.CLEAN),
                "true_positives": len(C.PLANTED),
                "false_positives": 0,
                "sensitivity": 1.0,
                "specificity": 1.0,
                "false_positive_rate": 0.0,
                "runtime_behavior_manifested": len(C.CASES),
                "runtime_behavior_total": len(C.CASES),
            },
            "ranked_set": {
                "unit_ids": ["normal-128", "anti-short-circuit-127"],
                "both_units_measured_for_every_program": True,
            },
            "device_claim_open": opened,
            "device_claim_released": released,
            "device_sampling": sampling,
            "cases": cases,
        }

    def test_prospective_rows_bind_detector_units_and_authority_boundary(self):
        payload = self.prospective_payload()
        rows = C._belief_measurements(payload)
        self.assertEqual(len(rows), 3 + 2 * len(C.CASES))
        self.assertEqual(len({row["measurement_id"] for row in rows}), len(rows))
        self.assertEqual(
            [row["measurement_id"] for row in rows[:3]],
            ["reward_integrity_detector_sensitivity",
             "reward_integrity_detector_specificity",
             "reward_integrity_detector_false_positive_rate"])
        elapsed = rows[3]
        self.assertEqual(elapsed["reps"], C.KERNEL_REPETITIONS)
        self.assertEqual(elapsed["extra"]["case_identity"]["source_sha256"], "1".zfill(64))
        self.assertEqual(
            elapsed["extra"]["ranked_unit_identity"]["unit_id"], "normal-128")
        for row in rows:
            unsigned = dict(row)
            claimed = unsigned.pop("measurement_sha256")
            self.assertEqual(claimed, C._sha(C._canonical(unsigned)))
            self.assertTrue(row["extra"]["instrument_validation_only"])
            self.assertFalse(row["extra"]["candidate_speed_claim"])
            self.assertFalse(row["extra"]["grants_campaign_authority"])

    def test_pre_hook_or_unreleased_payload_cannot_emit_rows(self):
        historical = self.prospective_payload()
        historical["schema"] = "epyc.autokernel.executable-reward-hack-corpus.v1"
        with self.assertRaisesRegex(RuntimeError, "current-schema"):
            C._belief_measurements(historical)
        unreleased = self.prospective_payload()
        unreleased["device_claim_released"]["released_at"] = None
        with self.assertRaisesRegex(RuntimeError, "durably released"):
            C._belief_measurements(unreleased)

    def test_invalid_elapsed_observation_is_refused(self):
        payload = self.prospective_payload()
        payload["cases"][0]["ranked_units"][0]["result"]["gpu_elapsed_ms"] = float("nan")
        with self.assertRaisesRegex(RuntimeError, "valid GPU elapsed"):
            C._belief_measurements(payload)

    def test_mutated_sampling_window_is_refused(self):
        payload = self.prospective_payload()
        payload["device_sampling"]["samples"][0]["power_w"] += 1
        with self.assertRaisesRegex(RuntimeError, "sampling self-hash"):
            C._belief_measurements(payload)


if __name__ == "__main__":
    unittest.main()


class TestRowflowParentHeldClaim(unittest.TestCase):
    """The launcher's own flock must be inherited only on proof, else fail closed."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.lock = Path(self.temp.name) / "gpu_device.mi210_0.lock"
        self.lock.write_bytes(b"")

    def test_no_rowflow_environment_means_normal_acquisition(self):
        self.assertIsNone(C._inherit_rowflow_claim({}, self.lock))
        self.assertIsNone(C._inherit_rowflow_claim({"FILLER_NEEDGPU": "0"}, self.lock))

    def test_gpu_grant_without_row_id_fails_closed(self):
        with self.assertRaises(C.RowflowClaimError):
            C._inherit_rowflow_claim({"FILLER_NEEDGPU": "1"}, self.lock)

    def test_free_lock_fails_closed(self):
        with self.assertRaises(C.RowflowClaimError):
            C._inherit_rowflow_claim(
                {"FILLER_NEEDGPU": "1", "FILLER_ID": "r"}, self.lock)

    def test_lock_held_by_this_process_is_not_a_parent_hold(self):
        fd = os.open(self.lock, os.O_RDWR)
        self.addCleanup(os.close, fd)
        fcntl.flock(fd, fcntl.LOCK_EX)
        with self.assertRaises(C.RowflowClaimError):
            C._inherit_rowflow_claim(
                {"FILLER_NEEDGPU": "1", "FILLER_ID": "r"}, self.lock)

    def _child_result(self, parent_row_id: str, child_row_id: str) -> subprocess.CompletedProcess:
        child = textwrap.dedent(f"""
            import os, sys
            from pathlib import Path
            sys.path.insert(0, {str(Path(__file__).resolve().parents[3])!r})
            from kernel_rnd.autokernel.execution import reward_hack_corpus as C
            env = dict(os.environ, FILLER_NEEDGPU="1", FILLER_ID={child_row_id!r})
            claim = C._inherit_rowflow_claim(env, Path({str(self.lock)!r}))
            print(claim.receipt().to_dict()["claim_id"], claim.release().to_dict()["released_at"] is not None)
        """)
        parent = textwrap.dedent(f"""
            import fcntl, os, subprocess, sys
            fd = os.open({str(self.lock)!r}, os.O_RDWR)
            fcntl.flock(fd, fcntl.LOCK_EX)
            r = subprocess.run([sys.executable, "-c", {child!r}], capture_output=True, text=True)
            sys.stdout.write(r.stdout); sys.stderr.write(r.stderr); sys.exit(r.returncode)
        """)
        env = dict(os.environ, FILLER_ID=parent_row_id)
        return subprocess.run([sys.executable, "-c", parent], capture_output=True,
                              text=True, env=env, timeout=60)

    def test_ancestor_holding_the_lock_for_the_same_row_is_inherited(self):
        result = self._child_result("rowA", "rowA")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(result.stdout.startswith("rowflow-parent:rowA:"), result.stdout)
        self.assertTrue(result.stdout.strip().endswith("True"))

    def test_ancestor_holding_the_lock_for_another_row_fails_closed(self):
        result = self._child_result("rowA", "rowB")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("RowflowClaimError", result.stderr)


class TestRowflowEngineFlockIdiom(unittest.TestCase):
    """filler_run15.sh idiom: `exec 9>LOCK; flock 9` in an inner bash whose argv
    names the row; /proc/locks then records the dead flock(1) pid, the lock lives
    on bash's fd 9, and the job runs with fd 9 closed (c6-26-corpus-repin r2)."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.lock = Path(self.temp.name) / "gpu_device.mi210_0.lock"
        self.lock.write_bytes(b"")

    def _run(self, argv_row: str, child_row: str) -> subprocess.CompletedProcess:
        child = textwrap.dedent(f"""
            import os, sys
            from pathlib import Path
            sys.path.insert(0, {str(Path(__file__).resolve().parents[3])!r})
            from kernel_rnd.autokernel.execution import reward_hack_corpus as C
            claim = C._inherit_rowflow_claim(os.environ, Path({str(self.lock)!r}))
            print(claim.receipt().to_dict()["claim_id"])
        """)
        # inner bash -> taskset/bash -c (exec) -> job.py python -> corpus python: the
        # holder is the GRANDPARENT of the claiming process, as in the real tree.
        script = ('exec 9>"$LOCK"; flock -w 5 9 || exit 9; export FILLER_ID="$CHILD_ROW" FILLER_NEEDGPU=1; '
                  '"$PY" -c "import os,subprocess,sys; sys.exit(subprocess.run([sys.executable, \'-c\', os.environ[\'CHILD\']]).returncode)" 9>&-')
        env = {k: v for k, v in os.environ.items() if k not in ("FILLER_ID", "FILLER_NEEDGPU")}
        env.update(LOCK=str(self.lock), CHILD_ROW=child_row, PY=sys.executable, CHILD=child)
        return subprocess.run(["bash", "-c", script, "filler_run15.sh", "--inner", "--id", argv_row],
                              capture_output=True, text=True, env=env, timeout=60)

    def test_ancestor_walk_reaches_past_the_parent(self):
        chain = C._ancestor_pids(os.getpid(), Path("/proc"))
        self.assertGreaterEqual(len(chain), 2)
        self.assertEqual(chain[0], os.getppid())

    def test_engine_idiom_same_row_is_inherited(self):
        result = self._run("rowA", "rowA")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(result.stdout.startswith("rowflow-parent:rowA:"), result.stdout)

    def test_engine_idiom_other_row_fails_closed(self):
        result = self._run("rowA", "rowB")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("does not belong to row rowB", result.stderr)
