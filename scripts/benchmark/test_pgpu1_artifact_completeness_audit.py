#!/usr/bin/env python3

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from pgpu1_artifact_completeness_audit import audit_artifact, audit_artifacts


def _write_artifact(path: Path, summary: dict, extra: str = "", identity: dict | None = None) -> None:
    path.mkdir(parents=True)
    (path / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if extra:
        (path / "notes.txt").write_text(extra)
    if identity is not None:
        (path / "binary_identity.json").write_text(json.dumps(identity, indent=2) + "\n")


# --- P-GPU-1 field-3 fixtures ------------------------------------------------
#
# Field 3 (binary/model identity) is a CONJUNCTION of four mandatory sub-fields,
# each of which must be matched in RECORDED RUN METADATA rather than anywhere in
# the artifact directory. The fixtures below exist so that re-breaking either
# defence (the conjunction, or the metadata scoping) fails a collected test.
# See docs/reviews/gpu-linkage-retro-certification-20260812.md §3.2a-bis.

_PROD_BINARY = "/mnt/raid0/llm/llama.cpp/build-hip/bin/llama-server"
_EXPERIMENTAL_BINARY = "/mnt/raid0/llm/llama.cpp-experimental/build-v8-hip/bin/llama-server"

# The exact shape of the false positive: a harness SOURCE file that SETS the
# variable. It must never satisfy `ld_library_path_value`.
_HARNESS_SOURCE = (
    "env = dict(os.environ)\n"
    'env["LD_LIBRARY_PATH"] = f"{GPU_D}:/opt/rocm/lib:" + env.get("LD_LIBRARY_PATH", "")\n'
    'cmd = ["taskset", "-c", "184-191", f"{GPU_D}/llama-server", "-m", arm["model"]]\n'
)


def _identity(binary: str = _PROD_BINARY, **overrides) -> dict:
    record = {
        "binary": binary,
        "environment": {"LD_LIBRARY_PATH": "/mnt/raid0/llm/llama.cpp/build-hip/bin:/opt/rocm/lib"},
        "devices": {
            "argv": [binary, "--list-devices"],
            "stdout": "Available devices:\n  ROCm0: AMD Instinct MI210 (65520 MiB, 65416 MiB free)\n",
        },
        "git": {
            "commit": {
                "argv": ["git", "rev-parse", "HEAD"],
                "stdout": "0db32c06e3e550065b78311a6031ef3dd2c4f27c\n",
            }
        },
    }
    record.update(overrides)
    return record


def _full_summary() -> dict:
    """Every mandatory field except the four field-3 sub-fields."""
    return {
        "status": "ok",
        "cleanup_process_blockers": [],
        "cpu_stack": "production stack quiesced by explicit interference policy",
        "warmup": "no warm-up; first rep discarded after graph recapture",
        "results": [
            {
                "rep": 5,
                "prompt_tps": 100.0,
                "decode_tps": 120.0,
                "median": 120.0,
                "mad": 1.0,
                "draft_n_accepted": 767,
                "model": "/mnt/raid0/llm/models/Qwen3.6-27B-MTP-Q8_0.gguf",
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
        "kernel": {
            "branch": "production-consolidated-v9",
            "production_named_kernel": True,
        },
    }


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
    """(a) A real recorded LD_LIBRARY_PATH value in run metadata satisfies field 3."""
    with tempfile.TemporaryDirectory() as tmp:
        artifact = Path(tmp) / "complete"
        _write_artifact(
            artifact,
            _full_summary(),
            extra="post-cleanup VRAM 0% and post-cleanup KFD none.",
            identity=_identity(),
        )

        result = audit_artifact(artifact)

    assert result["status"] == "complete"
    assert result["recommendation"] == "retro_cert_candidate"
    assert result["retro_cert_eligible"] is True
    assert result["disqualifications"] == []
    assert result["missing_required_fields"] == []
    field3 = result["field_results"]["binary_model_identity"]
    assert field3["state"] == "present"
    assert field3["missing_subfields"] == []
    assert sorted(field3["subfields"]) == [
        "backend_device_list",
        "binary_path",
        "kernel_commit",
        "ld_library_path_value",
    ]


def test_large_valid_summary_is_scanned_as_marker_and_cannot_be_vacuously_complete() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        artifact = Path(tmp) / "large_summary"
        _write_artifact(
            artifact,
            {"status": "ok", "padding": "x" * (2 * 1024 * 1024)},
        )

        result = audit_artifact(artifact, max_bytes=2 * 1024 * 1024)

    summary_path = str(artifact / "summary.json")
    assert result["summary_status"] == "ok"
    assert result["status"] == "incomplete"
    assert "summary_json" in result["present_required_fields"]
    assert "summary_json" not in result["missing_required_fields"]
    assert "result_grammar" in result["missing_required_fields"]
    assert result["files_scanned"] == [summary_path]
    assert result["files_skipped_large"] == [summary_path]


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


def test_field3_rejects_ld_library_path_mentioned_only_in_harness_source() -> None:
    """(b) THE regression that matters.

    The artifact bank no LD_LIBRARY_PATH value; its only hit for the string is a
    harness source file that SETS the variable. Before the 2026-08-12 fix this
    graded field 3 `present` on that hit alone.
    """
    with tempfile.TemporaryDirectory() as tmp:
        artifact = Path(tmp) / "harness_only_ldlib"
        identity = _identity()
        identity["environment"] = {"ROCR_VISIBLE_DEVICES": "0"}  # no LD_LIBRARY_PATH banked
        _write_artifact(artifact, _full_summary(), identity=identity)
        (artifact / "harness.py").write_text(_HARNESS_SOURCE)

        result = audit_artifact(artifact)

    field3 = result["field_results"]["binary_model_identity"]
    assert field3["state"] == "missing"
    assert field3["missing_subfields"] == ["ld_library_path_value"]
    assert field3["subfields"]["ld_library_path_value"]["matched_patterns"] == []
    # The other three sub-fields still match, so the failure is attributable.
    assert field3["subfields"]["backend_device_list"]["state"] == "present"
    assert field3["subfields"]["binary_path"]["state"] == "present"
    assert field3["subfields"]["kernel_commit"]["state"] == "present"
    assert "binary_model_identity" in result["missing_required_fields"]
    assert result["status"] == "incomplete"
    assert result["recommendation"] == "rerun_required"
    assert result["retro_cert_eligible"] is False
    # Scoping is the mechanism: the harness file was read, but not as metadata.
    assert str(artifact / "harness.py") in result["files_scanned"]
    assert str(artifact / "harness.py") not in result["run_metadata_files"]


def test_field3_rejects_value_shaped_text_that_lives_in_source_not_metadata() -> None:
    """(b2) Isolates the SCOPING defence.

    Here the harness source and the README carry text that looks exactly like a
    banked value (`LD_LIBRARY_PATH=/mnt/...`). Only the run-metadata restriction
    rejects it: shipping the harness must never satisfy the field.
    """
    with tempfile.TemporaryDirectory() as tmp:
        artifact = Path(tmp) / "value_shaped_source"
        identity = _identity()
        identity["environment"] = {"ROCR_VISIBLE_DEVICES": "0"}  # nothing banked
        _write_artifact(artifact, _full_summary(), identity=identity)
        (artifact / "harness.py").write_text(
            'LAUNCH = "LD_LIBRARY_PATH=/mnt/raid0/llm/llama.cpp/build-hip/bin llama-server"\n'
        )
        (artifact / "README.md").write_text(
            "Reproduce with `LD_LIBRARY_PATH=/mnt/raid0/llm/llama.cpp/build-hip/bin ...`\n"
        )

        result = audit_artifact(artifact)

    field3 = result["field_results"]["binary_model_identity"]
    assert field3["scope"] == "run_metadata"
    assert field3["missing_subfields"] == ["ld_library_path_value"]
    assert field3["state"] == "missing"
    assert result["recommendation"] == "rerun_required"
    for name in ("harness.py", "README.md"):
        assert str(artifact / name) in result["files_scanned"]
        assert str(artifact / name) not in result["run_metadata_files"]


def test_field3_rejects_bare_variable_name_even_inside_run_metadata() -> None:
    """The second, independent defence: a mention is not a value."""
    with tempfile.TemporaryDirectory() as tmp:
        artifact = Path(tmp) / "name_without_value"
        identity = _identity()
        identity["environment"] = {"note": "the launcher sets LD_LIBRARY_PATH before exec"}
        _write_artifact(artifact, _full_summary(), identity=identity)

        result = audit_artifact(artifact)

    field3 = result["field_results"]["binary_model_identity"]
    assert field3["state"] == "missing"
    assert field3["missing_subfields"] == ["ld_library_path_value"]


def test_field3_is_a_conjunction_every_subfield_is_load_bearing() -> None:
    """Mutation test: delete each sub-field's evidence in turn; each must fail alone."""
    mutations = {
        "ld_library_path_value": lambda rec: rec.update({"environment": {}}),
        "backend_device_list": lambda rec: rec.update({"devices": {"argv": [], "stdout": ""}}),
        "binary_path": lambda rec: rec.update(
            {
                "binary": "llama_server_binary_elided",
                # the --list-devices probe also carries the path; both must go
                "devices": {"argv": ["--list-devices"], "stdout": "Available devices:\n  ROCm0: MI210\n"},
            }
        ),
        "kernel_commit": lambda rec: rec.update({"git": {}}),
    }
    for subfield, mutate in mutations.items():
        with tempfile.TemporaryDirectory() as tmp:
            artifact = Path(tmp) / f"missing_{subfield}"
            identity = _identity()
            mutate(identity)
            _write_artifact(artifact, _full_summary(), identity=identity)

            result = audit_artifact(artifact)

        field3 = result["field_results"]["binary_model_identity"]
        assert field3["missing_subfields"] == [subfield], (
            f"removing {subfield} evidence must fail exactly that sub-field, "
            f"got {field3['missing_subfields']}"
        )
        assert field3["state"] == "missing"
        assert result["recommendation"] == "rerun_required"


def test_experimental_kernel_artifact_is_disqualified_with_a_stated_reason() -> None:
    """(c) Provenance rule: an experimental-kernel run can never be retro-certified."""
    with tempfile.TemporaryDirectory() as tmp:
        artifact = Path(tmp) / "experimental_but_complete"
        identity = _identity(binary=_EXPERIMENTAL_BINARY)
        identity["environment"] = {
            "LD_LIBRARY_PATH": "/mnt/raid0/llm/llama.cpp-experimental/build-v8-hip/bin:/opt/rocm/lib"
        }
        _write_artifact(artifact, _full_summary(), identity=identity)

        result = audit_artifact(artifact)

    # Fields are complete — the refusal is provenance, not completeness.
    assert result["status"] == "complete"
    assert result["missing_required_fields"] == []
    assert result["recommendation"] == "retro_cert_disqualified"
    assert result["retro_cert_eligible"] is False
    assert len(result["disqualifications"]) == 1
    record = result["disqualifications"][0]
    assert record["kernel_tree"] == "llama.cpp-experimental"
    assert "OBSERVATION-ONLY" in record["reason"]
    assert "gpu-cross-device.md" in record["reason"]


def test_guard_hygiene_probe_of_experimental_tree_does_not_disqualify() -> None:
    """The disqualifier must not fire on a mention that is not a build/binary tree.

    The v9 GPU certification run banks `git -C /mnt/raid0/llm/llama.cpp-experimental
    rev-parse HEAD` as a guard-hygiene side probe while measuring a production
    binary. Treating that as provenance would falsely refuse the artifact the
    production-v9 promotion attestation rests on.
    """
    with tempfile.TemporaryDirectory() as tmp:
        artifact = Path(tmp) / "prod_with_experimental_probe"
        identity = _identity()
        identity["experimental_head"] = {
            "argv": ["git", "-C", "/mnt/raid0/llm/llama.cpp-experimental", "rev-parse", "--short", "HEAD"],
            "stdout": "0db32c06e\n",
        }
        _write_artifact(artifact, _full_summary(), identity=identity)

        result = audit_artifact(artifact)

    assert result["disqualifications"] == []
    assert result["recommendation"] == "retro_cert_candidate"
    assert result["retro_cert_eligible"] is True


def test_batch_report_surfaces_disqualification_reasons() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        experimental = Path(tmp) / "experimental"
        identity = _identity(binary=_EXPERIMENTAL_BINARY)
        identity["environment"] = {
            "LD_LIBRARY_PATH": "/mnt/raid0/llm/llama.cpp-experimental/build-v8-hip/bin:/opt/rocm/lib"
        }
        _write_artifact(experimental, _full_summary(), identity=identity)

        report = audit_artifacts([experimental])

    assert report["recommendation"] == "rerun_required_for_incomplete_artifacts"
    assert report["disqualified_artifacts"] == [str(experimental)]
    assert report["disqualification_reasons"][0]["kernel_tree"] == "llama.cpp-experimental"


if __name__ == "__main__":
    raise SystemExit(
        "run under pytest: python3 -m pytest scripts/benchmark/test_pgpu1_artifact_completeness_audit.py"
    )
