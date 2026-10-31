import json
import tempfile
import unittest
import sys
from pathlib import Path

D = Path(__file__).resolve().parent
sys.path.insert(0, str(D.parent / "scripts/kernel_rnd"))
import replay_hip_cuid as r


class ReplayTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=D)
        self.root = Path(self.tmp.name)
        self.s = self.root / "source"
        self.b = self.root / "build"
        self.s.mkdir()
        self.b.mkdir()
        self.tu = self.s / "a.hip"
        self.tu.write_text("fixture TU\n")
        self.cfg = self.root / "cfg"
        self.cfg.write_text("pinned config\n")
        self.m = self.root / "recipe.json"
        self.row = {
            "compiler": sys.executable,
            "compiler_sha256": r.file_sha(Path(sys.executable).resolve()),
            "config_files": {str(self.cfg): r.file_sha(self.cfg)},
            "tu": {"a.hip": {"cuid": "1234", "source_sha256": r.file_sha(self.tu)}},
        }
        self.save()
        self.argv = [sys.executable, "-x", "hip", "-c", str(self.tu), "-o", str(self.b / "a.o")]

    def tearDown(self):
        self.tmp.cleanup()

    def save(self):
        self.m.write_text(json.dumps(self.row))
        self.h = r.file_sha(self.m)

    def prepare(self, argv=None):
        return r.prepare(self.m, self.h, self.s, self.b, argv or self.argv)

    def test_per_TU_literal_transport_output_relocation(self):
        first, p = self.prepare()
        self.argv[-1] = str(self.b / "relocated.o")
        second, q = self.prepare()
        self.assertEqual(first[1], "-cuid=1234")
        self.assertEqual(first[1], second[1])
        self.assertNotEqual(p["output"], q["output"])

    def test_manifest_tamper_refuses(self):
        self.m.write_text("{}")
        self.assertRaisesRegex(ValueError, "manifest changed", self.prepare)

    def test_source_change_refuses(self):
        self.tu.write_text("changed")
        self.assertRaisesRegex(ValueError, "TU bytes", self.prepare)

    def test_configuration_change_refuses(self):
        self.cfg.write_text("changed")
        self.assertRaisesRegex(ValueError, "configuration", self.prepare)

    def test_compiler_bytes_identity_refuses(self):
        self.row["compiler_sha256"] = "0" * 64
        self.save()
        self.assertRaisesRegex(ValueError, "compiler identity", self.prepare)

    def test_unknown_TU_refuses(self):
        p = self.s / "other.hip"
        p.write_text("fixture TU\n")
        self.argv[-3] = str(p)
        self.assertRaisesRegex(ValueError, "unreviewed TU", self.prepare)

    def test_output_escape_refuses(self):
        self.argv[-1] = str(self.root / "foreign.o")
        self.assertRaisesRegex(ValueError, "output outside", self.prepare)

    def test_symlink_source_escape_refuses(self):
        p = self.root / "foreign.hip"
        p.write_text("fixture TU\n")
        self.tu.unlink()
        self.tu.symlink_to(p)
        self.assertRaisesRegex(ValueError, "TU outside", self.prepare)

    def test_duplicate_global_nonce_refuses(self):
        self.row["tu"]["b.hip"] = dict(self.row["tu"]["a.hip"])
        self.save()
        self.assertRaisesRegex(ValueError, "distinct", self.prepare)

    def test_unreviewed_explicit_nonce_and_response_refuse(self):
        for flag in ["-cuid=guessed", "-fuse-cuid=random", "-fhip-cuid=random", "@hidden.rsp"]:
            with self.subTest(flag=flag):
                self.assertRaisesRegex(
                    ValueError, "override/response", self.prepare, [*self.argv, flag]
                )

    def test_forwarded_frontend_CUID_overrides_refuse(self):
        for flag in ["-Xclang", "-Xclang=-cuid=bad", "-Xarch_host"]:
            with self.subTest(flag=flag):
                self.assertRaisesRegex(
                    ValueError, "forwarding/configuration", self.prepare, [*self.argv, flag]
                )

    def test_compiler_configuration_overrides_refuse(self):
        for flag in [
            "--config",
            "--config=other.cfg",
            "--config-user-dir=other",
            "--config-system-dir=other",
            "--no-default-config",
        ]:
            with self.subTest(flag=flag):
                self.assertRaisesRegex(
                    ValueError, "forwarding/configuration", self.prepare, [*self.argv, flag]
                )

    def test_internal_frontend_modes_refuse(self):
        for flag in ["-cc1", "-cc1as"]:
            with self.subTest(flag=flag):
                self.assertRaisesRegex(
                    ValueError, "forwarding/configuration", self.prepare, [*self.argv, flag]
                )

    def test_cmake_dependency_values_are_not_source_inputs(self):
        extra = [
            "-MD",
            "-MT",
            "CMakeFiles/probe.dir/a.hip.o",
            "-MQ",
            str(self.root / "target-label"),
        ]
        for flags in [extra + ["-MF", str(self.b / "a.d")], extra + ["-MF" + str(self.b / "a.d")]]:
            with self.subTest(flags=flags):
                command, _ = self.prepare([*self.argv, *flags])
                self.assertEqual(command, [self.argv[0], "-cuid=1234", *self.argv[1:], *flags])

    def test_joined_dependency_target_labels_are_not_output_paths(self):
        command, _ = self.prepare(
            [
                *self.argv,
                "-MD",
                "-MToutside/target",
                "-MQoutside/quoted-target",
                "-MF" + str(self.b / "a.d"),
            ]
        )
        self.assertIn("-MToutside/target", command)

    def test_dependency_output_escapes_and_stdout_refuse(self):
        for value in [str(self.root / "foreign.d"), "../foreign.d", "-", str(self.b)]:
            for flags in [["-MF", value], ["-MF" + value]]:
                with self.subTest(flags=flags):
                    self.assertRaisesRegex(
                        ValueError,
                        "dependency output outside",
                        r.prepare,
                        self.m,
                        self.h,
                        self.s,
                        self.b,
                        [*self.argv, *flags],
                        self.b,
                    )

    def test_dependency_symlink_escape_refuses(self):
        (self.b / "foreign").symlink_to(self.root, target_is_directory=True)
        self.assertRaisesRegex(
            ValueError,
            "dependency output outside",
            self.prepare,
            [*self.argv, "-MF", str(self.b / "foreign/a.d")],
        )

    def test_dependency_option_missing_value_refuses(self):
        for flag in ["-MF", "-MT", "-MQ"]:
            with self.subTest(flag=flag):
                self.assertRaisesRegex(
                    ValueError, "missing option value", self.prepare, [*self.argv, flag]
                )


if __name__ == "__main__":
    unittest.main()
