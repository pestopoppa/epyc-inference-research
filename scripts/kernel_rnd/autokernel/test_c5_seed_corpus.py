#!/usr/bin/env python3
"""Deterministic tests for the HyRA C5 gfx90a reference registry."""

from __future__ import annotations

import copy
import hashlib
import json
import unittest

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
        self.assertNotIn("sol_score", json.dumps(task))
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


if __name__ == "__main__":
    unittest.main()
