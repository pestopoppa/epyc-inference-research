#!/usr/bin/env python3
import hashlib
import json
from pathlib import Path
import re

root = Path(__file__).parent

def read_probe(label):
    directory = root / label
    stdout = (directory / "stdout.txt").read_bytes()
    stderr = (directory / "stderr.txt").read_bytes()
    return {
        "label": label,
        "exit_code": int((directory / "exit_code.txt").read_text().strip()),
        "binary_sha256": (directory / "binary.sha256").read_text().split()[0],
        "stdout_sha256": hashlib.sha256(stdout).hexdigest(),
        "stderr_sha256": hashlib.sha256(stderr).hexdigest(),
        "combined_sha256": hashlib.sha256(stdout + b"\0" + stderr).hexdigest(),
    }, stdout, stderr

def normalize(data):
    return re.sub(
        rb"/mnt/raid0/llm/(?:llama\.cpp|llama\.cpp-experimental/build-v8-(?:cpu|hip)|tmp/v8-exact-tip-v7-hip-quant-selection-20260725T182000Z/build)/[^\n]*test-quant-type-selection",
        b"<test-quant-type-selection>",
        data,
    )

def compare(name, v7_label, v8_label):
    left, left_out, left_err = read_probe(v7_label)
    right, right_out, right_err = read_probe(v8_label)
    raw_identical = left_out == right_out and left_err == right_err
    normalized_left = normalize(left_out) + b"\0" + normalize(left_err)
    normalized_right = normalize(right_out) + b"\0" + normalize(right_err)
    normalized_identical = normalized_left == normalized_right
    (root / f"{name}-v7-normalized-output.bin").write_bytes(normalized_left)
    (root / f"{name}-v8-normalized-output.bin").write_bytes(normalized_right)
    return {
        "v7": left,
        "v8": right,
        "raw_byte_identical_stdout_and_stderr": raw_identical,
        "path_normalized_byte_identical_stdout_and_stderr": normalized_identical,
        "both_expected_exit_1": left["exit_code"] == 1 and right["exit_code"] == 1,
        "classification": "inherited_baseline_path_normalized" if normalized_identical and left["exit_code"] == 1 and right["exit_code"] == 1 else "unexpected_difference",
    }

cpu = compare("cpu", "v7-cpu", "v8-cpu")
hip = compare("hip", "v7-hip-scratch-retry", "v8-hip")
report = {
    "schema": "epyc.kernel_v8.exact_tip_quant_selection_differential.v2",
    "v7_source_head": (root / "v7_source_head.txt").read_text().strip(),
    "v8_source_head": (root / "v8_source_head.txt").read_text().strip(),
    "canonical_v7_hip_binary_path": "/mnt/raid0/llm/llama.cpp/build-hip/bin/test-quant-type-selection",
    "canonical_v7_hip_binary_available": False,
    "v7_hip_substitute": {
        "kind": "isolated_detached_worktree_build",
        "source_head": (root / "v7-hip-scratch-retry/source_head.txt").read_text().strip(),
        "reason": "The frozen canonical HIP runtime tree did not include the test executable. The isolated exact-v7 build used matching HIP CMake options and was removed after capture.",
    },
    "comparisons": {"cpu": cpu, "hip": hip},
    "overall_classification": "inherited_baseline_path_normalized" if cpu["classification"] == hip["classification"] == "inherited_baseline_path_normalized" else "unexpected_difference",
    "raw_difference_explanation": "The only raw CPU/HIP output difference is the final regenerate instruction, which embeds the absolute executable path. Raw output is retained; path-normalized output is the functional comparison.",
    "promotion_interpretation": "The nonzero quant-selection result is inherited at the exact v8 candidate tip under the documented path-normalized functional comparison. This is an inherited-baseline classification, not a passing quant-selection test.",
}
(root / "summary.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
