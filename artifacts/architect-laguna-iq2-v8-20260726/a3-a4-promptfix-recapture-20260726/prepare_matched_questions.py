#!/usr/bin/env python3
"""Materialize the exact terminal Laguna promptfix questions for A3/A4."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
LAGUNA_PROMPTFIX = Path(
    "/mnt/raid0/llm/epyc-inference-research/artifacts/architect-laguna-iq2-v8-20260726/"
    "scorer-artifact-rescore-20260726/clean-full40-promptfix-20260726/questions_pinned_40.json"
)
QUESTION_SHA256 = "4b03ad7703bbf2dbaa1eb91b3313cc3cab2892672db87f6242ffd1d489e76375"
RUNNER_SHA256 = "79721927e95293d070aba294bf422a24b1182dde07310d461d9e3ddaf6c84b0e"
WATCHDOG_SHA256 = "f4bd45b9617ca880a92be506d741038df65d457f0923f07bc3db7091a7303055"
BINARY_PATH = "/mnt/raid0/llm/llama.cpp/build-hip/bin/llama-server"
BINARY_SHA256 = "112c560f1c978c584a9899539851348a0ce1e05cde458061c281758aff066882"
KERNEL = "production-consolidated-v8"
KERNEL_HEAD = "67a433bf45a8a091d83b4ea0b32ff0735fd51800"
A3_MODEL_PATH = "/mnt/raid0/llm/models/Qwen3.6-27B-MTP-Q8_0.gguf"
A3_MODEL_SHA256 = "9408dcb356cc061a05c139e5647cbde0698ff980c6a69f7fc214e9989f86cfa8"
A4_MODEL_PATH = "/mnt/raid0/llm/models/Qwen3.6-35B-A3B-MTP-Q8_0.gguf"
A4_MODEL_SHA256 = "93dd505d5b4d3f6adcef8c3b6b35465f7537379893f80b87b9ddc2baa62ca557"


def atomic_write(path: Path, payload: bytes) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def main() -> int:
    raw = LAGUNA_PROMPTFIX.read_bytes()
    if hashlib.sha256(raw).hexdigest() != QUESTION_SHA256:
        raise RuntimeError("terminal Laguna promptfix question SHA mismatch")
    rows = json.loads(raw)
    ids = [row.get("id") for row in rows]
    if len(rows) != 40 or len(set(ids)) != 40 or any(not isinstance(item, str) for item in ids):
        raise RuntimeError("terminal Laguna promptfix question set is not 40 unique ordered IDs")
    atomic_write(HERE / "questions_pinned_40.json", raw)
    atomic_write(HERE / "expected_question_ids.json", (json.dumps(ids, indent=2) + "\n").encode())
    manifest = {
        "schema": "a3_a4_matched_promptfix_recapture.v1",
        "status": "PREPARED_NO_INFERENCE_EXECUTED",
        "question_source": str(LAGUNA_PROMPTFIX),
        "question_sha256": QUESTION_SHA256,
        "runner_source_sha256": RUNNER_SHA256,
        "watchdog_source_sha256": WATCHDOG_SHA256,
        "binary": {"path": BINARY_PATH, "sha256": BINARY_SHA256},
        "kernel": {"branch": KERNEL, "head": KERNEL_HEAD},
        "arms": {
            "A3_27B_dense": {
                "model": A3_MODEL_PATH,
                "model_sha256": A3_MODEL_SHA256,
                "hash_evidence": (
                    "/mnt/raid0/llm/epyc-inference-research/artifacts/architect-laguna-iq2-v8-20260726/"
                    "a3-a4-swe-confirmation/A3_27B_dense-port18091/model.sha256"
                ),
                "arm": "A3_27B_dense_v8_matched_laguna_promptfix_3072",
            },
            "A4_35B_A3B": {
                "model": A4_MODEL_PATH,
                "model_sha256": A4_MODEL_SHA256,
                "hash_evidence": (
                    "/mnt/raid0/llm/epyc-inference-research/data/kernel-v8-candidate/"
                    "cpu-prefill-regression/run-20260725T082414Z-v3-live/summary.partial.json#"
                    "/model_inventory/qwen36_q8/shards/0/sha256"
                ),
                "arm": "A4_35B_A3B_v8_matched_laguna_promptfix_3072",
            },
        },
        "sampling": {
            "seed": 42,
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,
            "max_tokens": 3072,
            "concurrency": 1,
            "repeats": 1,
            "endpoint": "chat",
            "enable_thinking": False,
        },
        "execution_gate": "requires verified 27B continuation.complete before --execute",
    }
    atomic_write(HERE / "prepared_manifest.json", (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode())
    print(json.dumps({"status": manifest["status"], "questions": 40, "sha256": QUESTION_SHA256}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
