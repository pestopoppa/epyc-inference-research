#!/usr/bin/env python3
"""Deterministic tests for the HyRA C5 gfx90a reference registry."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path

from . import c5_seed_corpus as C


class C5SeedCorpusTest(unittest.TestCase):
    def document(self):
        return json.loads(C._registry_path().read_text(encoding="utf-8"))

    def parse(self, document):
        return C.C5SeedCorpus.from_dict(document, registry_sha256="a" * 64)

    def test_checked_in_registry_pins_exact_seed_partition_and_provenance(self):
        corpus = C.load()
        self.assertEqual(tuple(seed.seed_id for seed in corpus.seeds), C.EXPECTED_SEED_IDS)
        self.assertEqual(
            {seed.seed_id for seed in corpus.seeds if seed.reference_kind == C.DIRECT_TRITON},
            C.DIRECT_TRITON_IDS,
        )
        self.assertEqual(
            {seed.seed_id for seed in corpus.seeds if seed.reference_kind == C.CUDA_REAUTHOR},
            C.CUDA_REAUTHOR_IDS,
        )
        self.assertEqual(corpus.source_revision, "26ebfbe7d491e6521d8bb5fc21fe88bb31460825")
        self.assertEqual(
            corpus.registry_sha256,
            hashlib.sha256(C._registry_path().read_bytes()).hexdigest(),
        )
        (evidence,) = corpus.policy_evidence
        self.assertEqual(evidence["evidence_id"], C.SOL_BOUND_EVIDENCE_ID)
        self.assertEqual(
            evidence["path"],
            "/workspace/handoffs/active/agentic-rocm-kernel-authoring.md",
        )
        self.assertEqual(
            evidence["sha256"],
            "c8cec57941b5c0954cd65b44719b984612d9c25094fce3e2ef4bcd42e8ec4f70",
        )

    def test_sol_bound_labels_claims_and_current_frame_eligibility_are_exact(self):
        corpus = C.load()
        by_id = {seed.seed_id: seed for seed in corpus.seeds}
        self.assertEqual(
            {seed_id: seed.bound_quality for seed_id, seed in by_id.items()},
            C.EXPECTED_BOUND_QUALITY,
        )
        self.assertEqual(
            {seed_id: seed.median_headroom_t_b_over_t_sol
             for seed_id, seed in by_id.items()},
            C.EXPECTED_HEADROOM,
        )
        self.assertEqual(by_id["k175"].traffic_basis, "defective_declared_traffic")
        self.assertNotIn(
            "source_sol_score_with_bound_quality_label", by_id["k175"].allowed_claims
        )
        for seed_id in ("k154", "k225", "k227", "k228"):
            self.assertNotIn(
                "source_sol_score_with_bound_quality_label", by_id[seed_id].allowed_claims
            )
        for seed in corpus.seeds:
            self.assertTrue(seed.current_frame_correctness_eligible)
            self.assertFalse(seed.current_frame_sol_score_eligible)

    def test_task_surface_is_gfx90a_bound_and_omits_nvidia_performance_numbers(self):
        corpus = C.load()
        (seed,) = corpus.select(("k215",))
        task = seed.task_descriptor(revision=corpus.source_revision)
        self.assertEqual(task["target"]["architecture"], "gfx90a")
        self.assertEqual(task["target"]["attestation_status"], "absent")
        self.assertEqual(task["reference"]["source_use"], "behavior_and_target_only")
        self.assertEqual(
            task["reference"]["observed_nvidia_bindings"],
            ["cublaslt", "cupy", "flashinfer", "wmma"],
        )
        self.assertEqual(
            task["sol_bound_policy"]["current_frame_eligibility"][
                "speed_of_light_objective"],
            "disabled_missing_gfx90a_constants")
        self.assertNotIn("sol_score", json.dumps(task, sort_keys=True))
        self.assertNotIn("median_headroom_t_b_over_t_sol", json.dumps(task))
        self.assertNotIn("latency_ms", json.dumps(task))

    def test_context_selection_is_ordered_hash_bound_and_reference_only(self):
        first = C.seed_context_item(("k228", "k138"))
        second = C.seed_context_item(("k228", "k138"))
        self.assertEqual(first, second)
        payload = json.loads(first.content)
        self.assertEqual(payload["authority"], "reference_only")
        self.assertEqual(
            [row["task_id"] for row in payload["tasks"]],
            ["hyra-sol-execbench/k228", "hyra-sol-execbench/k138"],
        )
        self.assertEqual(
            payload["policy_evidence"],
            [{
                "evidence_id": C.SOL_BOUND_EVIDENCE_ID,
                "path": "/workspace/handoffs/active/agentic-rocm-kernel-authoring.md",
                "sha256": "c8cec57941b5c0954cd65b44719b984612d9c25094fce3e2ef4bcd42e8ec4f70",
            }],
        )
        self.assertNotIn("33.4", first.content)
        self.assertNotIn("36837", first.content)
        self.assertTrue(first.source_ref.startswith(f"hyra-c5://{C.load().registry_sha256}/"))

    def test_unknown_duplicate_and_empty_selections_refuse(self):
        corpus = C.load()
        for selected in ((), ("k138", "k138"), ("k999",)):
            with self.subTest(selected=selected), self.assertRaises(C.SeedCorpusError):
                corpus.select(selected)

    def test_nvidia_attestation_cannot_be_relabelled_as_gfx90a_evidence(self):
        document = self.document()
        document["source_attestation"]["scope"] = "cross_vendor"
        with self.assertRaisesRegex(C.SeedCorpusError, "NVIDIA/Hopper-only"):
            self.parse(document)
        document = self.document()
        document["gfx90a_contract"]["attestation_status"] = "passed"
        with self.assertRaisesRegex(C.SeedCorpusError, "fresh MI210"):
            self.parse(document)

    def test_cuda_bound_target_must_name_its_nvidia_binding(self):
        document = self.document()
        row = next(row for row in document["seeds"] if row["seed_id"] == "k225")
        row["observed_nvidia_bindings"] = []
        with self.assertRaisesRegex(C.SeedCorpusError, "must name a binding"):
            self.parse(document)

    def test_seed_partition_artifact_identity_and_disposition_are_fail_closed(self):
        mutations = []
        document = self.document()
        document["seeds"][0]["reference_kind"] = C.CUDA_REAUTHOR
        document["seeds"][0]["observed_nvidia_bindings"] = ["cuda"]
        document["seeds"][0]["gfx90a"]["source_use"] = "behavior_and_target_only"
        mutations.append((document, "partition drifted"))

        document = self.document()
        document["seeds"][0]["artifact"] = document["seeds"][1]["artifact"]
        mutations.append((document, "filename does not bind"))

        document = self.document()
        document["seeds"][0]["gfx90a"]["disposition"] = "port_unchanged"
        mutations.append((document, "re-author/re-attest"))

        for document, message in mutations:
            with self.subTest(message=message), self.assertRaisesRegex(C.SeedCorpusError, message):
                self.parse(copy.deepcopy(document))

    def test_bound_policy_tampering_is_refused_adversarially(self):
        mutations = []

        document = self.document()
        document["seeds"][0]["sol_bound"]["quality"] = "meaningful"
        mutations.append((document, "bound-quality label drifted"))

        document = self.document()
        document["seeds"][4]["sol_bound"]["median_headroom_t_b_over_t_sol"] = 68.0
        mutations.append((document, "bound headroom drifted"))

        document = self.document()
        document["seeds"][3]["sol_bound"]["traffic_basis"] = "not_flagged"
        mutations.append((document, "declared-traffic policy drifted"))

        document = self.document()
        document["seeds"][2]["sol_bound"]["allowed_claims"].append(
            "source_sol_score_with_bound_quality_label"
        )
        mutations.append((document, "allowed SOL claims"))

        document = self.document()
        document["seeds"][4]["sol_bound"]["current_frame_eligibility"]["sol_score"] = True
        mutations.append((document, "lacks measured constants"))

        document = self.document()
        document["seeds"][4]["sol_bound"]["current_frame_eligibility"][
            "correctness_oracle"
        ] = False
        mutations.append((document, "correctness oracle must stay eligible"))

        for document, message in mutations:
            with self.subTest(message=message), self.assertRaisesRegex(C.SeedCorpusError, message):
                self.parse(copy.deepcopy(document))

    def test_policy_evidence_path_hash_and_schema_are_fail_closed(self):
        document = self.document()
        document["policy_evidence"][0]["path"] = "relative/handoff.md"
        with self.assertRaisesRegex(C.SeedCorpusError, "absolute"):
            self.parse(document)

        document = self.document()
        document["policy_evidence"][0]["sha256"] = "0" * 63
        with self.assertRaisesRegex(C.SeedCorpusError, "SHA-256"):
            self.parse(document)

        document = self.document()
        document["seeds"][0]["sol_bound"]["unreviewed_override"] = True
        with self.assertRaisesRegex(C.SeedCorpusError, "keys differ"):
            self.parse(document)

    def test_runtime_policy_evidence_bytes_and_file_identity_are_verified(self):
        document = self.document()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            evidence = root / "authority.md"
            evidence.write_bytes(b"reviewed policy\n")
            document["policy_evidence"][0]["path"] = str(evidence)
            document["policy_evidence"][0]["sha256"] = hashlib.sha256(
                evidence.read_bytes()).hexdigest()
            registry = root / "registry.json"
            registry.write_text(json.dumps(document), encoding="utf-8")
            C.load(registry)

            evidence.write_bytes(b"tampered policy\n")
            with self.assertRaisesRegex(C.SeedCorpusError, "SHA-256 mismatch"):
                C.load(registry)

            document["policy_evidence"][0]["sha256"] = hashlib.sha256(
                evidence.read_bytes()).hexdigest()
            link = root / "authority-link.md"
            link.symlink_to(evidence)
            document["policy_evidence"][0]["path"] = str(link)
            registry.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaisesRegex(C.SeedCorpusError, "cannot open evidence"):
                C.load(registry)

            hardlink = root / "authority-hardlink.md"
            os.link(evidence, hardlink)
            document["policy_evidence"][0]["path"] = str(hardlink)
            registry.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaisesRegex(C.SeedCorpusError, "single-link"):
                C.load(registry)


if __name__ == "__main__":
    unittest.main()
