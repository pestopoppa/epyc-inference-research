#!/bin/bash
set -euo pipefail

readonly ROOT="/mnt/raid0/llm/epyc-inference-research/data/kernel-v8-candidate/exact-tip-quant-selection/run-20260725T181406Z-67a433bf4"
readonly CLEAN_PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/opt/rocm/bin"
readonly V7_ROOT="/mnt/raid0/llm/llama.cpp"
readonly V8_ROOT="/mnt/raid0/llm/llama.cpp-experimental"

run_probe() {
    local label="$1"
    local binary="$2"
    local dir="$ROOT/$label"
    mkdir -p "$dir"
    printf '%q ' env -i "PATH=$CLEAN_PATH" LANG=C LC_ALL=C "$binary" > "$dir/command.txt"
    printf '\n' >> "$dir/command.txt"
    if [[ ! -x "$binary" ]]; then
        printf 'missing executable: %s\n' "$binary" > "$dir/stderr.txt"
        : > "$dir/stdout.txt"
        printf '%s\n' '127' > "$dir/exit_code.txt"
        printf '%s\n' 'missing' > "$dir/binary.sha256"
        printf '%s\n' 'missing' > "$dir/status.txt"
        return
    fi
    sha256sum "$binary" > "$dir/binary.sha256"
    set +e
    env -i "PATH=$CLEAN_PATH" LANG=C LC_ALL=C "$binary" > "$dir/stdout.txt" 2> "$dir/stderr.txt"
    local status=$?
    set -e
    printf '%s\n' "$status" > "$dir/exit_code.txt"
    printf '%s\n' 'executed' > "$dir/status.txt"
}

git -C "$V7_ROOT" rev-parse HEAD > "$ROOT/v7_source_head.txt"
git -C "$V8_ROOT" rev-parse HEAD > "$ROOT/v8_source_head.txt"
printf '%s\n' "PATH=$CLEAN_PATH" > "$ROOT/clean_environment.txt"
printf '%s\n' 'LANG=C' >> "$ROOT/clean_environment.txt"
printf '%s\n' 'LC_ALL=C' >> "$ROOT/clean_environment.txt"

run_probe v7-cpu "$V7_ROOT/build/bin/test-quant-type-selection"
run_probe v8-cpu "$V8_ROOT/build-v8-cpu/bin/test-quant-type-selection"
run_probe v7-hip "$V7_ROOT/build-hip/bin/test-quant-type-selection"
run_probe v8-hip "$V8_ROOT/build-v8-hip/bin/test-quant-type-selection"

python3 - "$ROOT" <<'PY'
import hashlib
import json
import pathlib
import re
import sys

root = pathlib.Path(sys.argv[1])
def payload(label):
    d = root / label
    stdout = (d / "stdout.txt").read_bytes()
    stderr = (d / "stderr.txt").read_bytes()
    return {
        "label": label,
        "exit_code": int((d / "exit_code.txt").read_text().strip()),
        "binary_sha256": (d / "binary.sha256").read_text().split()[0],
        "status": (d / "status.txt").read_text().strip(),
        "stdout_sha256": hashlib.sha256(stdout).hexdigest(),
        "stderr_sha256": hashlib.sha256(stderr).hexdigest(),
        "combined_sha256": hashlib.sha256(stdout + b"\\0" + stderr).hexdigest(),
    }, stdout, stderr

def normalize_invocation_path(data):
    return re.sub(
        rb"/mnt/raid0/llm/(?:llama\.cpp|llama\.cpp-experimental/build-v8-cpu)/[^\\n]*test-quant-type-selection",
        b"<test-quant-type-selection>",
        data,
    )

rows = {}
blobs = {}
for label in ("v7-cpu", "v8-cpu", "v7-hip", "v8-hip"):
    rows[label], *blobs[label] = payload(label)

comparisons = {}
for runtime in ("cpu", "hip"):
    a, b = f"v7-{runtime}", f"v8-{runtime}"
    available = rows[a]["status"] == "executed" and rows[b]["status"] == "executed"
    raw_same = available and blobs[a][0] == blobs[b][0] and blobs[a][1] == blobs[b][1]
    normalized_same = available and normalize_invocation_path(blobs[a][0]) == normalize_invocation_path(blobs[b][0]) and normalize_invocation_path(blobs[a][1]) == normalize_invocation_path(blobs[b][1])
    if available:
        (root / f"{runtime}-v7-normalized-output.txt").write_bytes(normalize_invocation_path(blobs[a][0]) + b"\\0" + normalize_invocation_path(blobs[a][1]))
        (root / f"{runtime}-v8-normalized-output.txt").write_bytes(normalize_invocation_path(blobs[b][0]) + b"\\0" + normalize_invocation_path(blobs[b][1]))
    comparisons[runtime] = {
        "v7_label": a,
        "v8_label": b,
        "byte_identical_stdout_and_stderr": raw_same if available else None,
        "path_normalized_stdout_and_stderr_identical": normalized_same if available else None,
        "comparison_available": available,
        "expected_exit_codes_observed": rows[a]["exit_code"] == 1 and rows[b]["exit_code"] == 1,
        "classification": "inherited_baseline_path_normalized" if normalized_same and rows[a]["exit_code"] == 1 and rows[b]["exit_code"] == 1 else ("unavailable" if not available else "unexpected_difference"),
    }

report = {
    "schema": "epyc.kernel_v8.exact_tip_quant_selection_differential.v1",
    "v7_source_head": (root / "v7_source_head.txt").read_text().strip(),
    "v8_source_head": (root / "v8_source_head.txt").read_text().strip(),
    "expected": {"exit_code": 1, "same_output_within_build_class": True},
    "probes": rows,
    "comparisons": comparisons,
    "overall_classification": "inherited_baseline_path_normalized" if all(x["classification"] == "inherited_baseline_path_normalized" for x in comparisons.values()) else ("incomplete" if any(x["classification"] == "unavailable" for x in comparisons.values()) else "unexpected_difference"),
    "promotion_interpretation": "This differential does not pass the quant-selection test. Where the paired executable is available, it demonstrates whether the nonzero result is inherited byte-for-byte from production v7 at the exact candidate tip. Unavailable comparisons are explicit evidence gaps, not passing results.",
}
(root / "summary.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
PY
