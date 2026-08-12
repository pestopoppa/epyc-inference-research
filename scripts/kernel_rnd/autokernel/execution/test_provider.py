from __future__ import annotations

import hashlib
import subprocess
import tempfile
import unittest
from pathlib import Path

from . import provider as P


class ProviderIsolationTest(unittest.TestCase):
    def test_candidate_prefix_is_admitted(self):
        with tempfile.TemporaryDirectory(prefix="ak-provider-") as root:
            prefix = P.IsolatedProviderPrefix.create(root)
            self.assertEqual(prefix.path, str(Path(root).resolve()))
            self.assertEqual(prefix.child("rocm", "lib"), Path(root, "rocm", "lib"))

    def test_shared_rocm_usr_and_production_are_refused(self):
        for path in (
            "/opt/rocm", "/opt/rocm/lib", "/usr", "/usr/local/rocm",
            "/mnt/raid0/llm/llama.cpp", "/mnt/raid0/llm/llama.cpp/build",
            "/mnt/raid0/llm/whisper.cpp/provider", "/mnt/raid0/llm/qwentts.cpp",
        ):
            with self.subTest(path=path), self.assertRaises(P.ProviderIsolationError):
                P.IsolatedProviderPrefix.create(path)

    def test_ancestors_that_encompass_shared_or_production_roots_are_refused(self):
        for path in ("/opt", "/mnt/raid0/llm", "/workspace", "/workspace/repos"):
            with self.subTest(path=path), self.assertRaises(P.ProviderIsolationError):
                P.IsolatedProviderPrefix.create(path)

    def test_sibling_of_shared_rocm_is_not_a_prefix_match(self):
        prefix = P.IsolatedProviderPrefix.create("/opt/rocm-ak-candidate")
        self.assertEqual(prefix.path, "/opt/rocm-ak-candidate")

    def test_relative_root_and_child_escape_are_refused(self):
        with self.assertRaises(P.ProviderIsolationError):
            P.IsolatedProviderPrefix.create("relative/provider")
        with tempfile.TemporaryDirectory(prefix="ak-provider-") as root:
            prefix = P.IsolatedProviderPrefix.create(root)
            with self.assertRaises(P.ProviderIsolationError):
                prefix.child("..", "escape")

    def test_symlink_into_shared_prefix_is_refused(self):
        with tempfile.TemporaryDirectory(prefix="ak-provider-link-") as root:
            link = Path(root, "rocm")
            link.symlink_to("/opt/rocm", target_is_directory=True)
            with self.assertRaises(P.ProviderIsolationError):
                P.IsolatedProviderPrefix.create(str(link))


class SourceProviderBuildTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="ak-provider-source-")
        self.root = Path(self.temporary.name)
        self.source = self.root / "source"
        self.source.mkdir()
        subprocess.run(("git", "init", "-q", str(self.source)), check=True)
        subprocess.run(("git", "-C", str(self.source), "config", "user.name", "test"),
                       check=True)
        subprocess.run(("git", "-C", str(self.source), "config", "user.email",
                        "test@example.invalid"), check=True)
        (self.source / "LICENSE").write_text("fixture license\n", encoding="utf-8")
        (self.source / "source.cpp").write_text("int provider_fixture;\n", encoding="utf-8")
        (self.source / "toolchain.json").write_text(
            '{"compiler":"fixture"}\n', encoding="utf-8")
        (self.source / "linkage.json").write_text(
            '{"shared_rocm":false}\n', encoding="utf-8")
        subprocess.run(("git", "-C", str(self.source), "add", "."), check=True)
        subprocess.run(("git", "-C", str(self.source), "commit", "-qm", "fixture"),
                       check=True)
        self.commit = subprocess.run(
            ("git", "-C", str(self.source), "rev-parse", "HEAD"), text=True,
            capture_output=True, check=True).stdout.strip()
        self.prefix = self.root / "provider-prefix"
        self.receipts = self.root / "receipts"
        self.receipts.mkdir()

    def tearDown(self):
        self.temporary.cleanup()

    @staticmethod
    def sha(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def reference(self, **updates):
        value = {
            "schema": "epyc.autokernel.provider_reference.v1",
            "kind": "rocm_library", "source_mode": "source",
            "source_ref": "fixture/provider", "source_commit": self.commit,
            "artifact_sha256": P.source_artifact_sha256(str(self.source)),
            "license_check": "recorded-for-provenance",
            "isolation_root": str(self.prefix),
            "toolchain_manifest_sha256": self.sha(self.source / "toolchain.json"),
            "linkage_manifest_sha256": self.sha(self.source / "linkage.json"),
            "target_backend": "llama_gpu",
            "evidence_authority": "candidate_eligible",
        }
        value.update(updates)
        return value

    def compile(self, reference=None):
        return P.compile_source_build(
            reference or self.reference(), source_root=str(self.source),
            argv=("/bin/sh", "-c",
                  'mkdir -p "$1/lib" && printf provider > "$1/lib/libprovider.so"',
                  "provider-build", str(self.prefix)),
            expected_outputs=("lib/libprovider.so",),
            toolchain_manifest=str(self.source / "toolchain.json"),
            linkage_manifest=str(self.source / "linkage.json"),
            license_file=str(self.source / "LICENSE"))

    def test_compile_binds_clean_source_manifests_and_isolated_prefix(self):
        plan = self.compile()
        document = plan.to_dict()
        self.assertEqual(document["provider_reference"]["source_commit"], self.commit)
        self.assertEqual(document["expected_outputs"], ["lib/libprovider.so"])
        self.assertFalse(document["network_allowed"])
        self.assertFalse(document["shared_rocm_mutation_allowed"])
        self.assertEqual(document["plan_sha256"], P._canonical_sha256({
            key: value for key, value in document.items() if key != "plan_sha256"}))

    def test_compile_refuses_opaque_diagnostic_or_non_llama_provider(self):
        cases = (
            {"source_mode": "opaque_binary", "source_commit": None,
             "evidence_authority": "diagnostic_only"},
            {"evidence_authority": "diagnostic_only"},
            {"target_backend": "llama_cpu"},
            {"kind": "llama_source"},
        )
        for updates in cases:
            with self.subTest(updates=updates), self.assertRaises(P.ProviderBuildError):
                self.compile(self.reference(**updates))

    def test_compile_refuses_source_drift_manifest_drift_and_prefix_escape(self):
        (self.source / "source.cpp").write_text("drift\n", encoding="utf-8")
        with self.assertRaisesRegex(P.ProviderBuildError, "not clean"):
            self.compile()
        subprocess.run(("git", "-C", str(self.source), "restore", "source.cpp"), check=True)
        with self.assertRaisesRegex(P.ProviderBuildError, "toolchain manifest"):
            self.compile(self.reference(toolchain_manifest_sha256="0" * 64))
        with self.assertRaises(P.ProviderBuildError):
            P.compile_source_build(
                self.reference(), source_root=str(self.source),
                argv=("/bin/true", str(self.prefix)), expected_outputs=("../escape",),
                toolchain_manifest=str(self.source / "toolchain.json"),
                linkage_manifest=str(self.source / "linkage.json"),
                license_file=str(self.source / "LICENSE"))

    def test_execute_requires_explicit_authorization(self):
        with self.assertRaisesRegex(P.ProviderBuildError, "explicit"):
            P.execute_source_build(self.compile(), receipt_root=str(self.receipts))

    def test_execute_builds_only_in_isolated_prefix_and_emits_receipt(self):
        receipt = P.execute_source_build(
            self.compile(), receipt_root=str(self.receipts), authorize_build=True,
            timeout_seconds=30)
        self.assertEqual(receipt["status"], "complete")
        self.assertEqual(receipt["outputs"][0]["path"], "lib/libprovider.so")
        self.assertEqual((self.prefix / "lib/libprovider.so").read_text(), "provider")
        self.assertFalse(receipt["network_accessed"])
        self.assertFalse(receipt["shared_rocm_mutated"])
        self.assertFalse(receipt["candidate_bankable_without_llama_gpu_integration"])
        self.assertTrue(receipt["teardown"]["verified_empty"])
        self.assertTrue(receipt["teardown"]["removed"])
        self.assertEqual(receipt["receipt_sha256"], P._canonical_sha256({
            key: value for key, value in receipt.items() if key != "receipt_sha256"}))

    def test_missing_expected_output_is_a_durable_failed_receipt(self):
        plan = P.compile_source_build(
            self.reference(), source_root=str(self.source),
            argv=("/bin/true", str(self.prefix)),
            expected_outputs=("lib/missing.so",),
            toolchain_manifest=str(self.source / "toolchain.json"),
            linkage_manifest=str(self.source / "linkage.json"),
            license_file=str(self.source / "LICENSE"))
        receipt = P.execute_source_build(
            plan, receipt_root=str(self.receipts), authorize_build=True,
            timeout_seconds=30)
        self.assertEqual(receipt["status"], "failed")
        self.assertEqual(receipt["outputs"], [])
        self.assertIn("omitted expected output", receipt["output_errors"][0])
        self.assertTrue(receipt["teardown"]["verified_empty"])


if __name__ == "__main__":
    unittest.main()
