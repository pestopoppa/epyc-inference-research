import subprocess
from pathlib import Path
import tempfile
import unittest

from . import microbench
from . import powercap_broker as P


CID = "a" * 64


def completed(argv, *, rc=0, out="", err=""):
    return subprocess.CompletedProcess(argv, rc, stdout=out, stderr=err)


class ScriptedRunner:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def __call__(self, argv, **kwargs):
        self.calls.append(tuple(argv))
        if not self.responses:
            raise AssertionError(f"unexpected command: {argv}")
        response = self.responses.pop(0)
        return completed(argv, **response)


def starts_and_reads(*, rows="0\t100\t1000\t/powercap/intel-rapl:0/energy_uj\n"):
    return [
        {"out": P.DEFAULT_IMAGE_ID + "\n"},
        {"out": CID + "\n"},
        {"out": f"true {P.DEFAULT_IMAGE_ID}\n"},
        {"out": rows},
    ]


class PowercapBrokerTests(unittest.TestCase):

    def test_reader_is_pinned_networkless_read_only_and_names_exact_container(self):
        with tempfile.TemporaryDirectory() as tmp:
            runner = ScriptedRunner(starts_and_reads() + [
                {"out": CID + "\n"},
                {"rc": 1, "err": f"Error: No such object: {CID}"},
            ])
            broker = P.PowercapBroker(powercap_root=tmp, runner=runner)
            counters = broker.read_package_energy(packages=(0,))
            receipt = broker.receipt()
            broker.close()

        run = runner.calls[1]
        self.assertIn("--network", run)
        self.assertIn("none", run)
        self.assertIn("--read-only", run)
        self.assertIn("--cap-drop", run)
        self.assertIn("ALL", run)
        self.assertIn("no-new-privileges", run)
        self.assertEqual(runner.calls[3][0:3], ("docker", "exec", CID))
        self.assertEqual(runner.calls[4], ("docker", "stop", "--time", "5", CID))
        self.assertEqual(counters[0][0:3], (0, 100, 1000))
        self.assertIn(CID, counters[0][3])
        self.assertIn(P.DEFAULT_IMAGE_ID, counters[0][3])
        self.assertEqual(receipt.container_id, CID)
        self.assertFalse(broker.active)
        self.assertEqual(runner.responses, [])

    def test_missing_requested_package_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            runner = ScriptedRunner(starts_and_reads() + [
                {"out": CID + "\n"},
                {"rc": 1, "err": f"Error: No such object: {CID}"},
            ])
            broker = P.PowercapBroker(powercap_root=tmp, runner=runner)
            try:
                with self.assertRaisesRegex(P.PowercapBrokerError, r"package.s. \[1\]"):
                    broker.read_package_energy(packages=(1,))
            finally:
                broker.close()
        self.assertEqual(runner.responses, [])

    def test_image_identity_mismatch_stops_before_container_creation(self):
        with tempfile.TemporaryDirectory() as tmp:
            runner = ScriptedRunner([{"out": "sha256:" + "b" * 64 + "\n"}])
            broker = P.PowercapBroker(powercap_root=tmp, runner=runner)
            with self.assertRaisesRegex(P.PowercapBrokerError, "image is unavailable"):
                broker.start()
        self.assertEqual(len(runner.calls), 1)

    def test_live_container_after_stop_is_killed_by_exact_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            runner = ScriptedRunner(starts_and_reads()[:3] + [
                {"out": CID + "\n"},
                {"out": "true\n"},
                {"out": CID + "\n"},
                {"rc": 1, "err": f"Error: No such object: {CID}"},
            ])
            broker = P.PowercapBroker(powercap_root=tmp, runner=runner)
            broker.start()
            broker.close()
        self.assertEqual(runner.calls[5], ("docker", "kill", CID))
        self.assertEqual(runner.responses, [])


class HostStateBrokerSeamTests(unittest.TestCase):

    def test_broker_supplies_only_energy_while_host_supplies_topology_and_clock(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sysfs = root / "sys"
            proc = root / "proc"
            for cpu in (0, 1):
                (sysfs / f"cpu{cpu}" / "cpufreq").mkdir(parents=True)
                (sysfs / f"cpu{cpu}" / "topology").mkdir()
                (sysfs / f"cpu{cpu}" / "cpufreq" / "scaling_cur_freq").write_text(
                    "2500000\n")
                (sysfs / f"cpu{cpu}" / "topology" / "physical_package_id").write_text(
                    "0\n")
            (sysfs / "cpu0" / "cpufreq" / "cpuinfo_min_freq").write_text("1000000\n")
            (sysfs / "cpu0" / "cpufreq" / "cpuinfo_max_freq").write_text("3500000\n")
            proc.mkdir()
            (proc / "loadavg").write_text("0.25 0.5 1.0 1/1 1\n")
            (proc / "uptime").write_text("1000.0 0.0\n")
            seen = []

            def energy(*, packages):
                seen.append(packages)
                return ((0, 123, 1000, "broker:test"),)

            state = microbench.read_host_state(
                cpu_list="0-1", sysfs_root=sysfs, proc_root=proc,
                package_energy_reader=energy, monotonic=lambda: 50.0)

        self.assertEqual(seen, [(0,)])
        self.assertEqual(state.khz_by_cpu, ((0, 2500000), (1, 2500000)))
        self.assertEqual(state.package_by_cpu, ((0, 0), (1, 0)))
        self.assertEqual(state.package_energy_uj, ((0, 123, 1000, "broker:test"),))

    def test_broker_failure_is_recorded_as_unreadable_not_passed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sysfs = root / "sys"
            proc = root / "proc"
            (sysfs / "cpu0" / "cpufreq").mkdir(parents=True)
            (sysfs / "cpu0" / "topology").mkdir()
            (sysfs / "cpu0" / "cpufreq" / "scaling_cur_freq").write_text("2500000\n")
            (sysfs / "cpu0" / "topology" / "physical_package_id").write_text("0\n")
            proc.mkdir()
            (proc / "loadavg").write_text("0.0 0.0 0.0 1/1 1\n")
            (proc / "uptime").write_text("1000.0 0.0\n")

            def broken(*, packages):
                raise P.PowercapBrokerError("reader died")

            state = microbench.read_host_state(
                cpu_list="0", sysfs_root=sysfs, proc_root=proc,
                package_energy_reader=broken, monotonic=lambda: 50.0)

        self.assertEqual(state.package_energy_uj, ())
        self.assertTrue(any("reader died" in item for item in state.unreadable))


if __name__ == "__main__":
    unittest.main()
