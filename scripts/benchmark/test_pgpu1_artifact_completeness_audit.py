#!/usr/bin/env python3

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from pgpu1_artifact_completeness_audit import audit_artifact, audit_artifacts


def _write_artifact(path: Path, summary: dict, extra: str = "") -> None:
    path.mkdir(parents=True)
    (path / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if extra:
        (path / "notes.txt").write_text(extra)


def test_primary_near_misses_do_not_satisfy_explicit_policy_fields() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        artifact = Path(tmp) / "gate_r_candidate"
        _write_artifact(
            artifact,
            {
                "status": "ok",
                "cleanup_process_blockers": [],
                "results": [
                    {
                        "rep": 5,
                        "prompt_tps": 100.0,
                        "decode_tps": 120.0,
                        "draft_n_accepted": 767,
                        "memory_samples": [
                            {
                                "phase": "before_cleanup",
                                "rocm": {
                                    "argv": [
                                        "rocm-smi",
                                        "--showpidgpus",
                                        "--showmemuse",
                                        "--showuse",
                                    ],
                                    "stdout": "GPU Memory Allocated (VRAM%): 50\nGPU use (%): 99",
                                },
                            }
                        ],
                        "cleanup": {"dead": True},
                    }
                ],
            },
            extra=(
                "llama-server /mnt/raid0/llm/llama.cpp-experimental rev-parse "
                "/mnt/raid0/llm/models/model.gguf LD_LIBRARY_PATH ROCm0 median MAD n=5 "
                "fresh-server reps"
            ),
        )

        result = audit_artifact(artifact)

    assert result["status"] == "incomplete"
    assert "cpu_interference_policy" in result["missing_required_fields"]
    assert "warmup_discard_policy" in result["missing_required_fields"]
    assert "production_named_kernel_identity" in result["missing_required_fields"]
    assert "post_cleanup_vram_sample" in result["missing_required_fields"]
    assert "cpu_interference_policy" in result["near_miss_fields"]
    assert "warmup_discard_policy" in result["near_miss_fields"]
    assert "production_named_kernel_identity" in result["near_miss_fields"]
    assert "post_cleanup_vram_sample" in result["near_miss_fields"]


def test_complete_artifact_is_retro_cert_candidate() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        artifact = Path(tmp) / "complete"
        _write_artifact(
            artifact,
            {
                "status": "ok",
                "cleanup_process_blockers": [],
                "results": [
                    {
                        "rep": 5,
                        "prompt_tps": 100.0,
                        "decode_tps": 120.0,
                        "draft_n_accepted": 767,
                        "memory_samples": [
                            {
                                "phase": "after_cleanup",
                                "rocm": {
                                    "argv": [
                                        "rocm-smi",
                                        "--showpidgpus",
                                        "--showmemuse",
                                        "--showuse",
                                        "--showclocks",
                                        "--showpower",
                                        "--showtemp",
                                    ],
                                    "stdout": (
                                        "GPU Memory Allocated (VRAM%): 0\nGPU use (%): 0\n"
                                        "sclk mclk power temperature no KFD PIDs"
                                    ),
                                },
                            }
                        ],
                        "cleanup": {"dead": True},
                    }
                ],
            },
            extra=(
                "llama-server /mnt/raid0/llm/llama.cpp rev-parse production-consolidated-v7 "
                "production_named_kernel: true "
                "/mnt/raid0/llm/models/model.gguf LD_LIBRARY_PATH ROCm0 median MAD n=5 "
                "no warm-up; no discard after graph recapture. CPU stack quiesced by policy. "
                "post-cleanup VRAM 0% and post-cleanup KFD none."
            ),
        )

        result = audit_artifact(artifact)

    assert result["status"] == "complete"
    assert result["recommendation"] == "retro_cert_candidate"
    assert result["missing_required_fields"] == []


def test_batch_report_marks_incomplete_when_any_artifact_fails() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        complete = Path(tmp) / "complete"
        incomplete = Path(tmp) / "incomplete"
        _write_artifact(
            complete,
            {"status": "ok", "results": [{"rep": 5, "prompt_tps": 1, "decode_tps": 1, "draft_n_accepted": 1, "cleanup": {"dead": True}}]},
            extra=(
                "summary.json rocm-smi --showpidgpus --showmemuse --showuse --showclocks --showpower --showtemp "
                "llama-server production-consolidated-v7 production_named_kernel: true rev-parse model.gguf LD_LIBRARY_PATH ROCm0 median MAD n=5 "
                "no warm-up discard graph recapture CPU stack quiesced after_cleanup no KFD PIDs 0% VRAM"
            ),
        )
        _write_artifact(incomplete, {"status": "ok", "results": []})

        report = audit_artifacts([complete, incomplete])

    assert report["status"] == "incomplete"
    assert report["recommendation"] == "rerun_required_for_incomplete_artifacts"


if __name__ == "__main__":
    test_primary_near_misses_do_not_satisfy_explicit_policy_fields()
    test_complete_artifact_is_retro_cert_candidate()
    test_batch_report_marks_incomplete_when_any_artifact_fails()
