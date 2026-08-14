from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import tempfile
from types import MappingProxyType
import unittest

from .. import schemas
from . import gpu_load_admission as A
from .split_runtime_verifier import HotResidencyIdentity


H = lambda letter: letter * 64


def profile() -> dict:
    return {
        "profile_id": "mi210-qwen-tg128-v1",
        "model_path": "/models/qwen.gguf", "model_sha256": H("a"),
        "model_bytes": 400_000_000, "workload": "tg128",
        "calls_per_arm": 9, "device_id": "mi210_0",
        "cold_load_host_bytes": 400_000_000,
        "worst_case_loads_per_interval": 18,
        "minimum_headroom_bytes_per_s": 2_000_000_000,
        "telemetry_max_age_ms": 2_000, "evidence_sha256": H("b"),
    }


def example(identifier: str, polarity: str) -> dict:
    positive = polarity == "positive"
    return {
        "id": identifier, "polarity": polarity,
        "facts": {"profile_id": "mi210-qwen-tg128-v1",
                  "telemetry": "complete" if positive else "missing"},
        "missing": [] if positive else ["headroom"],
        "mode": "cold_overlap" if positive else "cold_serialized",
        "rationale": "all exact facts" if positive else "missing observation",
        "disqualifiers": [] if positive else ["telemetry_missing"],
        "counterfactual": "serialize if facts differ" if positive else "observe headroom",
        "evidence": ["sha256:" + H("c" if positive else "d")],
    }


def corpus_body() -> dict:
    value = {
        "schema": A.POLICY_SCHEMA, "version": "site-mi210-v1",
        "profiles": [profile()],
        "examples": [example("exact-overlap", "positive"),
                     example("missing-headroom", "negative")],
    }
    value["policy_sha256"] = schemas.content_hash(value)
    return value


def write_corpus(root: Path, value: dict | None = None) -> tuple[Path, str]:
    path = root / "policy.json"
    path.write_text(json.dumps(value or corpus_body(), sort_keys=True))
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def request(**changes) -> A.AdmissionRequest:
    values = {
        "model_path": "/models/qwen.gguf", "model_sha256": H("a"),
        "model_bytes": 400_000_000, "workload": "tg128",
        "calls_per_arm": 9, "device_id": "mi210_0",
        "cold_load_host_bytes": 400_000_000,
        "worst_case_loads_per_interval": 18,
        "telemetry_observed": True, "telemetry_age_ms": 100,
        "observed_headroom_bytes_per_s": 3_000_000_000,
        "telemetry_receipt_sha256": H("e"),
    }
    values.update(changes)
    return A.AdmissionRequest(**values)


def hot_identity(model_path: Path) -> HotResidencyIdentity:
    body = {
        "schema": "epyc.autokernel.split_reward_runtime_maps.v1",
        "runtime_manifest_sha256": H("1"), "arm": "candidate",
        "reward_binary_sha256": H("2"), "hip_library_sha256": H("3"),
        "model_path": str(model_path), "model_sha256": H("a"),
        "device_id": "mi210_0", "kfd_pid": 123, "boot_id": "boot-a",
        "process_start_ticks": 456,
        "mapped_local_sha256": {"/runtime/libggml-hip.so.0": H("3")},
    }
    return HotResidencyIdentity(
        runtime_manifest_sha256=H("1"), arm="candidate",
        reward_binary_sha256=H("2"), hip_library_sha256=H("3"),
        model_path=model_path, model_sha256=H("a"),
        device_id="mi210_0", kfd_pid=123, boot_id="boot-a",
        process_start_ticks=456,
        mapped_local_sha256=MappingProxyType(body["mapped_local_sha256"]),
        identity_sha256=schemas.content_hash(body))


class GpuLoadAdmissionTests(unittest.TestCase):
    def load(self, root: Path, value: dict | None = None) -> A.PolicyCorpus:
        path, digest = write_corpus(root, value)
        return A.load_policy_corpus(path, expected_file_sha256=digest)

    def test_policy_parser_seals_deep_positive_and_negative_corpus(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            policy = self.load(Path(directory))
        self.assertEqual(policy.version, "site-mi210-v1")
        self.assertEqual({row.polarity for row in policy.examples},
                         {"positive", "negative"})
        self.assertIsInstance(policy.sealed, MappingProxyType)
        self.assertIsInstance(policy.examples[0].facts, MappingProxyType)
        with self.assertRaises(TypeError):
            policy.examples[0].facts["telemetry"] = "changed"

    def test_policy_refuses_outer_or_inner_hash_tamper_and_bad_examples(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, digest = write_corpus(root)
            with self.assertRaisesRegex(A.AdmissionPolicyError, "sealed digest"):
                A.load_policy_corpus(path, expected_file_sha256=H("0"))
            link = root / "policy-link.json"; link.symlink_to(path.name)
            with self.assertRaisesRegex(A.AdmissionPolicyError, "non-symlink"):
                A.load_policy_corpus(link, expected_file_sha256=digest)
            value = corpus_body(); value["version"] = "changed-v1"
            path, digest = write_corpus(root, value)
            with self.assertRaisesRegex(A.AdmissionPolicyError, "self-hash"):
                A.load_policy_corpus(path, expected_file_sha256=digest)
            for mutation in ("positive_missing", "duplicate", "unknown_key"):
                value = corpus_body()
                if mutation == "positive_missing":
                    value["examples"] = [example("one", "positive")]
                elif mutation == "duplicate":
                    value["examples"][1]["id"] = value["examples"][0]["id"]
                else:
                    value["examples"][0]["argv"] = ["unsafe"]
                value["policy_sha256"] = schemas.content_hash(
                    {key: item for key, item in value.items()
                     if key != "policy_sha256"})
                path, digest = write_corpus(root, value)
                with self.subTest(mutation=mutation), \
                        self.assertRaises(A.AdmissionPolicyError):
                    A.load_policy_corpus(path, expected_file_sha256=digest)

    def test_exact_profile_and_explicit_headroom_authorize_overlap(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            policy = self.load(Path(directory))
        decision = A.arbitrate(policy, request(), actor_recommendation="cold_overlap")
        self.assertEqual(decision.mode, "cold_overlap")
        self.assertEqual(decision.profile["profile_id"], "mi210-qwen-tg128-v1")
        self.assertEqual(decision.disqualifiers, ())
        receipt = decision.to_dict()
        A.validate_decision_receipt(receipt)
        self.assertEqual(receipt["policy_version"], policy.version)
        self.assertEqual(receipt["policy_sha256"], policy.policy_sha256)
        self.assertEqual(receipt["policy_file_sha256"], policy.file_sha256)
        self.assertEqual(receipt["actor_recommendation"], "cold_overlap")

    def test_every_bound_profile_fact_and_telemetry_fail_closed(self) -> None:
        mutations = {
            "model_path": "/models/other.gguf", "model_sha256": H("f"),
            "model_bytes": 400_000_001, "workload": "pp512",
            "calls_per_arm": 8, "device_id": "mi300_0",
            "cold_load_host_bytes": 400_000_001,
            "worst_case_loads_per_interval": 19,
        }
        with tempfile.TemporaryDirectory() as directory:
            policy = self.load(Path(directory))
        for field, value in mutations.items():
            with self.subTest(field=field):
                self.assertEqual(A.arbitrate(
                    policy, request(**{field: value}),
                    actor_recommendation="cold_overlap").mode,
                    "cold_serialized")
        unsafe = (
            request(telemetry_observed=False, telemetry_age_ms=None,
                    observed_headroom_bytes_per_s=None,
                    telemetry_receipt_sha256=None),
            request(telemetry_age_ms=2_001),
            request(observed_headroom_bytes_per_s=1_999_999_999),
            request(foreign_kfd_pids=(999,)),
        )
        for item in unsafe:
            self.assertEqual(A.arbitrate(policy, item).mode, "cold_serialized")

    def test_exact_self_validating_hot_identity_authorizes_resident_mode(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            policy = self.load(root)
            model = root / "qwen.gguf"; model.write_bytes(b"model")
            hot = hot_identity(model.resolve())
            item = request(model_path=str(model.resolve()),
                           runtime_manifest_sha256=H("1"), runtime_arm="candidate",
                           hot_residency=hot,
                           expected_hot_identity_sha256=hot.identity_sha256,
                           residency_revalidated=True)
            self.assertEqual(A.arbitrate(policy, item).mode, "hot_resident")
            # A different expected identity cannot turn a cold request hot.
            item = replace(item, expected_hot_identity_sha256=H("0"))
            self.assertEqual(A.arbitrate(policy, item).mode, "cold_serialized")

    def test_actor_can_preserve_or_downgrade_but_never_upgrade(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            policy = self.load(Path(directory))
        safe = A.arbitrate(policy, request(),
                           actor_recommendation="cold_serialized")
        self.assertEqual(safe.mode, "cold_serialized")
        mismatch = request(workload="pp512")
        for recommendation in ("cold_overlap", "hot_resident"):
            decision = A.arbitrate(policy, mismatch,
                                   actor_recommendation=recommendation)
            self.assertEqual(decision.mode, "cold_serialized")
            self.assertIn("actor_recommendation_not_authoritative",
                          decision.disqualifiers)

    def test_decision_receipt_binds_every_authority_field(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            policy = self.load(Path(directory))
        receipt = A.arbitrate(policy, request()).to_dict()
        for field in ("policy_version", "policy_sha256", "policy_file_sha256",
                      "request", "profile",
                      "actor_recommendation", "mode", "reason", "disqualifiers"):
            changed = json.loads(json.dumps(receipt))
            if field in {"request", "profile"}:
                changed[field]["workload"] = "changed"
            elif field == "disqualifiers":
                changed[field] = ["changed"]
            elif field == "actor_recommendation":
                changed[field] = "cold_serialized"
            else:
                changed[field] = H("0") if "sha256" in field else "changed"
            with self.subTest(field=field), \
                    self.assertRaises(A.AdmissionPolicyError):
                A.validate_decision_receipt(changed)


if __name__ == "__main__":
    unittest.main()
