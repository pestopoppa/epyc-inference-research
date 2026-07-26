#!/usr/bin/env bash
# Collect fresh v4 ThinkingCap and Fable A/B arms.
# This script owns one sequential MI210 server lifecycle and requires --execute.
set -euo pipefail

REPO=/mnt/raid0/llm/epyc-inference-research
RUNROOT="$REPO/artifacts/architect-27b-finetunes-v8-20260726/live-20260726T1750Z"
OUT="$RUNROOT/continuation-27b-v8"
SAME="$REPO/artifacts/architect-same-era-v8-20260726/live-20260726T201413Z"
LAGUNA_BASE="$REPO/artifacts/architect-laguna-iq2-v8-20260726/scorer-artifact-rescore-20260726/clean-full40-promptfix-20260726"
LAGUNA_VALIDATOR="$LAGUNA_BASE/validate_clean_full40_capture.py"
LAGUNA_ABORT_RECEIPT="$LAGUNA_BASE/BASE_DIAGNOSTIC_SUPERSESSION_ABORT_RECEIPT.json"
LAGUNA_ARM=Laguna_S_2_1_UD_IQ2_M_v8_clean_full40_promptfix_3072
LAGUNA_QUESTION_SHA=4b03ad7703bbf2dbaa1eb91b3313cc3cab2892672db87f6242ffd1d489e76375
LAGUNA_VALIDATOR_SHA=511e77db440022596728d4887467e855c11b4fe7b076cd0a6de3d2f866085124
LAGUNA_ABORT_RECEIPT_SHA=471f71b5651169ee06a2fb5c7a18bf0a6a7ecd2a626d95aeaef61a79554a282d
MANIFEST="$RUNROOT/../finetune_bench_manifest.json"
SERVER=/mnt/raid0/llm/llama.cpp/build-hip/bin/llama-server
WATCHDOG="$REPO/scripts/benchmark/capture_integrity_watchdog.py"
SWEBENCH_VERIFIED_SOURCE="$REPO/artifacts/architect-code-eval-20260724/swebench_verified.json"
SWEBENCH_VERIFIED_SHA=b087b5dad72b3e765a6cf93a9e7d516d8796698a0fd358abb73c6627df19f66e
PRE_REPAIR_IDENTITY_SHA=6212109e18668e6c7f6ec488cbb573af4bea61ef90ec0ba67391578b4b30cbc2
PORT=18092
CORES=184-191
SERVER_PID=""
RUNNER_SHA=""
WATCHDOG_SHA=""
CONVERTER_SHA=""
LAGUNA_VALIDATION=""
HEALTH_TIMEOUT_S=180
LIVE_STATUS_TIMEOUT_S=300
STOP_TIMEOUT_S=30

die() { printf '27b-continuation: %s\n' "$*" >&2; return 1; }
port_listening() { lsof -nP -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; }

wait_for_exit() {
    local pid=$1 deadline=$((SECONDS + STOP_TIMEOUT_S))
    while kill -0 "$pid" 2>/dev/null; do
        (( SECONDS < deadline )) || return 1
        sleep 1
    done
}

stop_owned_server() {
    local pid=${SERVER_PID:-}
    [[ -n "$pid" ]] || return 0
    if kill -0 "$pid" 2>/dev/null; then
        kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
        if ! wait_for_exit "$pid"; then
            kill -KILL -- "-$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null || true
            wait_for_exit "$pid" || die "owned server PID $pid survived SIGKILL"
        fi
    fi
    wait "$pid" 2>/dev/null || true
    ! ps -p "$pid" >/dev/null 2>&1 || die "owned server PID $pid remains live"
    ! port_listening || die "port $PORT remains live after owned cleanup"
    SERVER_PID=""
}

cleanup() {
    local rc=$?
    set +e
    stop_owned_server
    exit "$rc"
}

wait_for_health() {
    local dir=$1 deadline=$((SECONDS + HEALTH_TIMEOUT_S))
    while (( SECONDS < deadline )); do
        if curl -fsS --max-time 5 "http://127.0.0.1:$PORT/health" >"$dir/health.json" 2>/dev/null; then
            return 0
        fi
        kill -0 "$SERVER_PID" 2>/dev/null || die "server exited before health"
        sleep 2
    done
    die "health timeout after ${HEALTH_TIMEOUT_S}s"
}

prove_listener() {
    local out=$1
    python3 - "$PORT" "$SERVER_PID" >"$out" <<'PY'
import json, os, pathlib, sys
port, pid = int(sys.argv[1]), int(sys.argv[2])
needle = f":{port:04X}"
rows = pathlib.Path("/proc/net/tcp").read_text().splitlines()[1:]
inodes = {row.split()[9] for row in rows if row.split()[3] == "0A" and row.split()[1].endswith(needle)}
fds = []
for entry in pathlib.Path(f"/proc/{pid}/fd").iterdir():
    try:
        target = os.readlink(entry)
    except OSError:
        continue
    if target in {f"socket:[{inode}]" for inode in inodes}:
        fds.append(entry.name)
if not inodes or not fds:
    raise SystemExit("listener ownership proof failed")
print(json.dumps({"pid": pid, "port": port, "socket_inodes": sorted(inodes), "pid_fds": sorted(fds)}, sort_keys=True))
PY
}

find_valid_clean_laguna() {
    python3 - "$MANIFEST" "$LAGUNA_BASE" "$LAGUNA_VALIDATOR" "$LAGUNA_ABORT_RECEIPT" \
        "$LAGUNA_ARM" "$LAGUNA_QUESTION_SHA" "$LAGUNA_VALIDATOR_SHA" \
        "$LAGUNA_ABORT_RECEIPT_SHA" <<'PY'
import hashlib
import json
import subprocess
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
base, validator, abort_receipt = map(Path, sys.argv[2:5])
arm, question_sha, validator_sha, abort_receipt_sha = sys.argv[5:9]
spec = manifest["execution_prerequisites"]["clean_laguna_full40"]
expected_spec = {
    "base": str(base),
    "validation_file": "capture.validation.json",
    "validator": str(validator),
    "validator_sha256": validator_sha,
    "expected_arm": arm,
    "question_source_sha256": question_sha,
    "supersession_abort_receipt": str(abort_receipt),
    "supersession_abort_receipt_sha256": abort_receipt_sha,
    "status": "VALID",
    "rows": 40,
    "capture_schema_version": "v7_quality_gate_capture.v4",
    "runner_source_sha256": "79721927e95293d070aba294bf422a24b1182dde07310d461d9e3ddaf6c84b0e",
}
if any(spec.get(key) != value for key, value in expected_spec.items()):
    raise SystemExit(1)
if (
    not validator.is_file()
    or hashlib.sha256(validator.read_bytes()).hexdigest() != validator_sha
    or not abort_receipt.is_file()
    or hashlib.sha256(abort_receipt.read_bytes()).hexdigest() != abort_receipt_sha
):
    raise SystemExit(1)
receipt = json.loads(abort_receipt.read_text(encoding="utf-8"))
if (
    receipt.get("status") != "ABORTED_SUPERSEDED_CLEAN"
    or receipt.get("replacement_arm") != arm
    or receipt.get("owned_processes_verified_dead") is not True
    or receipt.get("port_18089_listener_after_abort") is not False
):
    raise SystemExit(1)
expected = {
    "status": spec["status"],
    "rows": spec["rows"],
    "capture_schema_version": spec["capture_schema_version"],
    "runner_source_sha256": spec["runner_source_sha256"],
}
markers = sorted(
    base.glob(f"run-*/{spec['validation_file']}"),
    key=lambda path: path.stat().st_mtime_ns,
    reverse=True,
)
for marker in markers:
    try:
        recorded = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        continue
    if any(recorded.get(key) != value for key, value in expected.items()):
        continue
    checked = subprocess.run(
        [sys.executable, str(validator), str(marker.parent)],
        check=False,
        text=True,
        capture_output=True,
    )
    try:
        regenerated = json.loads(checked.stdout)
    except json.JSONDecodeError:
        continue
    if checked.returncode == 0 and regenerated == recorded:
        print(marker)
        raise SystemExit(0)
raise SystemExit(1)
PY
}

require_prerequisites() {
    test -f "$SAME/same_era_raw_runs.complete" || { die "same-era A4/A3 raw chain incomplete"; return 1; }
    LAGUNA_VALIDATION=$(find_valid_clean_laguna) || {
        die "no validator-confirmed clean Laguna full40 v4 capture exists"
        return 1
    }
    ! pgrep -af '[s]ame_era_v8_chain|[r]un_clean_full40.sh' >/dev/null || { die "upstream GPU chain remains active"; return 1; }
    ! port_listening || { die "port $PORT is occupied"; return 1; }
}

wait_for_prerequisites() {
    test -f "$SAME/same_era_raw_runs.complete" || die "same-era A4/A3 raw chain incomplete"
    while :; do
        if LAGUNA_VALIDATION=$(find_valid_clean_laguna); then
            if ! pgrep -af '[s]ame_era_v8_chain|[r]un_clean_full40.sh' >/dev/null; then
                ! port_listening || die "port $PORT is occupied"
                return 0
            fi
        elif ! pgrep -af '[r]un_clean_full40.sh' >/dev/null; then
            die "no valid clean Laguna full40 marker and no clean capture is active"
        fi
        printf '27b-continuation: waiting for terminal VALID clean Laguna full40 v4 capture\n'
        sleep 15
    done
}

prepare_output_dir() {
    if [[ -e "$OUT" ]]; then
        test ! -f "$OUT/continuation.complete" || die "continuation output is already complete: $OUT"
        local preserved="$OUT.superseded-incomplete-$(date -u +%Y%m%dT%H%M%SZ)"
        test ! -e "$preserved" || die "superseded output target exists: $preserved"
        mv "$OUT" "$preserved"
        printf '27b-continuation: preserved incomplete output at %s\n' "$preserved"
    fi
    mkdir -p "$OUT/instrument"
}

verify_frozen_inputs() {
    [[ $(sha256sum "$SWEBENCH_VERIFIED_SOURCE" | awk '{print $1}') == "$SWEBENCH_VERIFIED_SHA" ]] \
        || die "SWE-bench verified source drift"
    cp "$REPO/scripts/benchmark/v7_quality_gate_runner.py" "$OUT/instrument/v7_quality_gate_runner.py"
    cp "$WATCHDOG" "$OUT/instrument/capture_integrity_watchdog.py"
    cp "$REPO/artifacts/architect-code-eval-20260724/convert_sr_to_patch.py" "$OUT/instrument/convert_sr_to_patch.py"
    cp "$SWEBENCH_VERIFIED_SOURCE" "$OUT/instrument/swebench_verified.json"
    cp "$REPO/artifacts/architect-code-eval-20260724/questions_swebench_oracle.json" "$OUT/instrument/questions_swe_oracle.json"
    cp "$REPO/artifacts/architect-code-eval-20260724/questions_livecodebench_hard.json" "$OUT/instrument/questions_livecodebench_hard.json"
    RUNNER_SHA=$(sha256sum "$OUT/instrument/v7_quality_gate_runner.py" | awk '{print $1}')
    WATCHDOG_SHA=$(sha256sum "$OUT/instrument/capture_integrity_watchdog.py" | awk '{print $1}')
    CONVERTER_SHA=$(sha256sum "$OUT/instrument/convert_sr_to_patch.py" | awk '{print $1}')
    local fable_nm fable_mtp
    fable_nm=$(model_path fable_non_mtp)
    fable_mtp=$(model_path fable_mtp)
    /mnt/raid0/llm/llama.cpp/build-hip/bin/llama-gguf "$fable_nm" r n \
        >"$OUT/instrument/fable_non_mtp.gguf_header.txt"
    /mnt/raid0/llm/llama.cpp/build-hip/bin/llama-gguf "$fable_mtp" r n \
        >"$OUT/instrument/fable_mtp.gguf_header.txt"
    python3 - "$MANIFEST" "$SERVER" "$OUT/instrument/v7_quality_gate_runner.py" "$OUT/instrument/capture_integrity_watchdog.py" "$OUT/instrument/convert_sr_to_patch.py" "$OUT/instrument/swebench_verified.json" "$OUT/instrument/questions_swe_oracle.json" "$OUT/instrument/questions_livecodebench_hard.json" "$OUT/instrument/identity.json" "$OUT/instrument/fable_non_mtp.gguf_header.txt" "$OUT/instrument/fable_mtp.gguf_header.txt" "$LAGUNA_VALIDATION" <<'PY'
import hashlib, importlib.util, json, sys
from pathlib import Path
manifest, server, runner, watchdog, converter, swe_gold, swe_questions, lcb_questions, out, nm_header, mtp_header, laguna_validation = map(Path, sys.argv[1:])
data = json.loads(manifest.read_text())
def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()
spec = importlib.util.spec_from_file_location("finetune_prep", manifest.parent / "finetune_bench_runner.py")
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
witness = module.validate(data, headers=True)
expected_components = module.COMPONENT_BINDINGS
for role, snapshot in (("quality_runner", runner), ("capture_integrity_watchdog", watchdog), ("swe_converter", converter)):
    if sha(snapshot) != expected_components[role][1]:
        raise SystemExit(f"snapshot drift: {role}")
for suite, snapshot in (("swe_oracle", swe_questions), ("lcb_hard", lcb_questions)):
    if sha(snapshot) != data["inputs"][suite]["sha256"]:
        raise SystemExit(f"snapshot drift: {suite}")
identity = {"manifest_sha256": sha(manifest), "server_sha256": sha(server), "runner_sha256": sha(runner), "watchdog_sha256": sha(watchdog), "converter_sha256": sha(converter), "swebench_verified_sha256": sha(swe_gold), "capture_schema_version": data["capture_contract"]["schema_version"], "question_sha256": {"swe_oracle": sha(swe_questions), "lcb_hard": sha(lcb_questions)}, "full_static_witness": witness, "upstream_clean_laguna_validation": {"path": str(laguna_validation), "sha256": sha(laguna_validation), "payload": json.loads(laguna_validation.read_text())}, "models": {name: data["models"][name] for name in ("thinkingcap", "stock_non_mtp", "fable_non_mtp", "fable_mtp")}, "fable_header_contract": "validated_851_base_plus_15_mtp", "fable_header_transcripts": {"non_mtp_sha256": sha(nm_header), "mtp_sha256": sha(mtp_header)}}
tmp = out.with_suffix(".tmp"); tmp.write_text(json.dumps(identity, indent=2, sort_keys=True)+"\n"); tmp.replace(out)
PY
}

repair_instrument_for_resume() {
    test -d "$OUT/instrument" || die "resume output has no instrument bundle"
    test ! -f "$OUT/continuation.complete" || die "continuation is already complete"
    local identity="$OUT/instrument/identity.json"
    test -f "$identity" || die "resume instrument identity is missing"
    [[ $(sha256sum "$identity" | awk '{print $1}') == "$PRE_REPAIR_IDENTITY_SHA" ]] \
        || die "resume instrument identity is not the reviewed pre-repair identity"
    [[ $(sha256sum "$SWEBENCH_VERIFIED_SOURCE" | awk '{print $1}') == "$SWEBENCH_VERIFIED_SHA" ]] \
        || die "SWE-bench verified source drift"

    cp "$SWEBENCH_VERIFIED_SOURCE" "$OUT/instrument/swebench_verified.json"
    RUNNER_SHA=$(sha256sum "$OUT/instrument/v7_quality_gate_runner.py" | awk '{print $1}')
    WATCHDOG_SHA=$(sha256sum "$OUT/instrument/capture_integrity_watchdog.py" | awk '{print $1}')
    CONVERTER_SHA=$(sha256sum "$OUT/instrument/convert_sr_to_patch.py" | awk '{print $1}')

    local thinkingcap_dir="$OUT/A3-tc-quality__thinkingcap"
    local path
    for path in server.argv server.stdout server.stderr server.launch.json health.json listener.json; do
        if [[ -e "$thinkingcap_dir/$path" ]]; then
            test ! -e "$thinkingcap_dir/$path.pre-resume-swe" \
                || die "preserved pre-resume server evidence already exists: $path"
            mv "$thinkingcap_dir/$path" "$thinkingcap_dir/$path.pre-resume-swe"
        fi
    done
    for path in swe_oracle.converter.argv swe_oracle.converter.stdout swe_oracle.converter.stderr; do
        if [[ -e "$thinkingcap_dir/$path" ]]; then
            test ! -e "$thinkingcap_dir/$path.failed-missing-gold" \
                || die "preserved converter failure evidence already exists: $path"
            mv "$thinkingcap_dir/$path" "$thinkingcap_dir/$path.failed-missing-gold"
        fi
    done

    python3 - "$identity" "$OUT/instrument/swebench_verified.json" "$0" \
        "$PRE_REPAIR_IDENTITY_SHA" "$SWEBENCH_VERIFIED_SHA" \
        "$(git -C "$REPO" rev-parse HEAD)" <<'PY'
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

identity_path, gold_path, resume_script = map(Path, sys.argv[1:4])
pre_identity_sha, expected_gold_sha, research_head = sys.argv[4:7]

def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()

if sha(gold_path) != expected_gold_sha:
    raise SystemExit("copied SWE-bench verified source hash mismatch")
data = json.loads(identity_path.read_text())
data["swebench_verified_sha256"] = expected_gold_sha
data["instrument_repair"] = {
    "schema": "27b_continuation_instrument_repair.v1",
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "reason": "converter dependency omitted from sealed instrument bundle",
    "pre_repair_identity_sha256": pre_identity_sha,
    "resume_script_sha256": sha(resume_script),
    "research_head": research_head,
    "resume_policy": "reuse validator-clean 40/40 ThinkingCap SWE capture; do not redraw",
    "timing_exclusion": {
        "arm": "A3-tc-quality__thinkingcap",
        "suite": "swe_oracle",
        "question_id": "django__django-11239",
        "reason": "exogenous model hash read overlapped this request",
        "overlap_utc": ["2026-07-26T22:53:48Z", "2026-07-26T22:54:39Z"],
        "quality_response_valid": True,
        "decode_timing_eligible": False,
    },
}
tmp = identity_path.with_suffix(".tmp")
tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
tmp.replace(identity_path)
PY
}

validate_capture() {
    local pq=$1 summary=$2 questions=$3 expected=$4 source_sha=$5 suite=$6 arm=$7
    python3 - "$pq" "$summary" "$questions" "$expected" "$source_sha" "$suite" "$arm" <<'PY'
import hashlib, json, sys
from pathlib import Path
pq, summary, questions = map(Path, sys.argv[1:4])
expected, source_sha, expected_suite, expected_arm = int(sys.argv[4]), sys.argv[5], sys.argv[6], sys.argv[7]
question_rows = json.loads(questions.read_text())
rows = [json.loads(line) for line in pq.read_text().splitlines() if line.strip()]
expected_ids = [question["id"] for question in question_rows]
actual_ids = [row.get("id") for row in rows]
if (
    len(question_rows) != expected
    or len(set(expected_ids)) != expected
    or len(rows) != expected
    or actual_ids != expected_ids
    or len(set(actual_ids)) != expected
):
    raise SystemExit("capture is not the exact ordered pinned denominator")
for i, row in enumerate(rows):
    if row.get("request_error"):
        raise SystemExit(f"request error in row {i}: {row.get('request_error')}")
    if (
        row.get("capture_schema_version") != "v7_quality_gate_capture.v4"
        or row.get("runner_source_sha256") != source_sha
    ):
        raise SystemExit(f"row {i} lacks current v4/source identity")
    if (
        row.get("suite") != expected_suite
        or row.get("arm") != expected_arm
        or row.get("seed") != 42
        or row.get("rep") != 0
    ):
        raise SystemExit(f"row {i} draw identity drift")
    if row.get("prompt") != question_rows[i].get("prompt"):
        raise SystemExit(f"row {i} prompt differs from pinned question")
    for text_key, fingerprint_key in (
        ("prompt", "prompt_fingerprint"),
        ("response", "response_fingerprint"),
        ("reasoning", "reasoning_fingerprint"),
    ):
        text, fingerprint = row.get(text_key), row.get(fingerprint_key)
        if not isinstance(text, str) or not isinstance(fingerprint, dict):
            raise SystemExit(f"row {i} lacks {text_key} full-capture evidence")
        encoded = text.encode("utf-8")
        if fingerprint != {"chars": len(text), "utf8_bytes": len(encoded), "sha256": hashlib.sha256(encoded).hexdigest()}:
            raise SystemExit(f"row {i} has invalid {text_key} fingerprint")
data = json.loads(summary.read_text())
if len(data.get("suites", [])) != 1 or data["suites"][0].get("n") != expected:
    raise SystemExit("summary denominator drift")
live = pq.with_suffix(".live-status.json")
if (
    not live.is_file()
    or not (status := json.loads(live.read_text())).get("complete")
    or status.get("completed_draws") != expected
    or status.get("expected_draws") != expected
    or status.get("request_error_rows") != 0
    or status.get("artifact_integrity_fail_closed") is not False
    or status.get("schema_version") != "v7_quality_gate_capture.v4"
    or status.get("runner_source_sha256") != source_sha
    or status.get("suite") != expected_suite
    or status.get("arm") != expected_arm
):
    raise SystemExit("live capture contract missing or unhealthy")
PY
}

convert_swe_capture() {
    local arm_dir=$1 pq=$2 arm=$3
    local predictions="$arm_dir/swe_oracle.predictions.json"
    local diagnostics="$arm_dir/swe_oracle.predictions.diagnostics.jsonl"
    local diagnostic_summary="$arm_dir/swe_oracle.predictions.diagnostics.summary.json"
    local -a converter=(
        python3 -B "$OUT/instrument/convert_sr_to_patch.py"
        "$pq" "$arm" "$predictions"
        --runner-source "$OUT/instrument/v7_quality_gate_runner.py"
        --diagnostics-jsonl "$diagnostics"
        --diagnostics-summary "$diagnostic_summary"
    )
    printf '%q ' "${converter[@]}" >"$arm_dir/swe_oracle.converter.argv"
    printf '\n' >>"$arm_dir/swe_oracle.converter.argv"
    "${converter[@]}" >"$arm_dir/swe_oracle.converter.stdout" 2>"$arm_dir/swe_oracle.converter.stderr"
    python3 - "$pq" "$predictions" "$diagnostics" "$diagnostic_summary" "$RUNNER_SHA" <<'PY'
import json
import sys
from pathlib import Path

pq_path, predictions_path, diagnostics_path, summary_path = map(Path, sys.argv[1:5])
source_sha = sys.argv[5]
rows = [json.loads(line) for line in pq_path.read_text().splitlines() if line.strip()]
predictions = json.loads(predictions_path.read_text())
diagnostics = [json.loads(line) for line in diagnostics_path.read_text().splitlines() if line.strip()]
summary = json.loads(summary_path.read_text())
expected_ids = [row["id"] for row in rows]
if (
    len(predictions) != 40
    or [row.get("instance_id") for row in predictions] != expected_ids
    or len(diagnostics) != 40
    or [row.get("instance_id") for row in diagnostics] != expected_ids
):
    raise SystemExit("converter output is not the exact 40-row source denominator")
if (
    not summary.get("scoring_eligible")
    or not summary.get("prediction_artifact_written")
    or summary.get("prediction_count") != 40
    or summary.get("runner_source_sha256") != source_sha
    or summary.get("artifact_integrity_status") != "verified"
):
    raise SystemExit("converter output is not v4/source-bound and scoring-eligible")
prediction_by_id = {row["instance_id"]: row for row in predictions}
diagnostic_by_id = {row["instance_id"]: row for row in diagnostics}
length_ids = [row["id"] for row in rows if row.get("finish_reason") == "length"]
for instance_id in length_ids:
    if prediction_by_id[instance_id].get("model_patch") != "":
        raise SystemExit(f"length-finished row recovered a partial patch: {instance_id}")
    diagnostic = diagnostic_by_id[instance_id]
    if (
        diagnostic.get("conversion_disposition") != "model_truncation_empty_patch"
        or diagnostic.get("empty_patch") is not True
    ):
        raise SystemExit(f"length-finished row lacks terminal empty-failure evidence: {instance_id}")
expected_status = "terminal_model_length_failure" if length_ids else "complete"
if summary.get("conversion_status") != expected_status:
    raise SystemExit(
        f"converter disposition drift: {summary.get('conversion_status')} != {expected_status}"
    )
PY
}

wait_for_live_status() {
    # The runner publishes status after its first response, not at process start.
    # A 3K-4K token first draw can legitimately take well over 30 seconds.
    local status=$1 runner_pid=$2 deadline=$((SECONDS + LIVE_STATUS_TIMEOUT_S))
    while (( SECONDS < deadline )); do
        test -s "$status" && return 0
        kill -0 "$runner_pid" 2>/dev/null || die "runner exited before live-status startup"
        sleep 1
    done
    die "runner did not publish live status within ${LIVE_STATUS_TIMEOUT_S} seconds"
}

stop_child() {
    local pid=$1
    kill -0 "$pid" 2>/dev/null || return 0
    kill -TERM "$pid" 2>/dev/null || true
    for _ in $(seq 1 15); do
        if ! kill -0 "$pid" 2>/dev/null; then
            wait "$pid" 2>/dev/null || true
            ! ps -p "$pid" >/dev/null 2>&1 || die "child PID $pid remains after TERM"
            return 0
        fi
        sleep 1
    done
    kill -KILL "$pid" 2>/dev/null || true
    for _ in $(seq 1 5); do
        if ! kill -0 "$pid" 2>/dev/null; then
            wait "$pid" 2>/dev/null || true
            ! ps -p "$pid" >/dev/null 2>&1 || die "child PID $pid remains after SIGKILL"
            return 0
        fi
        sleep 1
    done
    die "child PID $pid survived SIGKILL"
}

await_runner_and_watchdog() {
    local runner_pid=$1 watchdog_pid=$2 runner_rc watchdog_rc
    while :; do
        local runner_alive=false watchdog_alive=false
        kill -0 "$runner_pid" 2>/dev/null && runner_alive=true
        kill -0 "$watchdog_pid" 2>/dev/null && watchdog_alive=true

        if [[ "$runner_alive" == true && "$watchdog_alive" == true ]]; then
            sleep 1
            continue
        fi

        if [[ "$runner_alive" == true && "$watchdog_alive" == false ]]; then
            if wait "$watchdog_pid"; then watchdog_rc=0; else watchdog_rc=$?; fi
            if (( watchdog_rc != 0 )); then
                stop_child "$runner_pid"
                wait "$runner_pid" 2>/dev/null || true
                return "$watchdog_rc"
            fi
            # A successful watcher has seen terminal status.  The runner may
            # still be serializing its summary; do not kill it.
            if wait "$runner_pid"; then runner_rc=0; else runner_rc=$?; fi
            return "$runner_rc"
        fi

        if [[ "$runner_alive" == false && "$watchdog_alive" == true ]]; then
            if wait "$runner_pid"; then runner_rc=0; else runner_rc=$?; fi
            if (( runner_rc != 0 )); then
                stop_child "$watchdog_pid"
                wait "$watchdog_pid" 2>/dev/null || true
                return "$runner_rc"
            fi
            # A successful runner still requires a successful observer.
            if wait "$watchdog_pid"; then watchdog_rc=0; else watchdog_rc=$?; fi
            return "$watchdog_rc"
        fi

        if wait "$runner_pid"; then runner_rc=0; else runner_rc=$?; fi
        if wait "$watchdog_pid"; then watchdog_rc=0; else watchdog_rc=$?; fi
        if (( runner_rc != 0 )); then
            return "$runner_rc"
        fi
        return "$watchdog_rc"
    done
}

model_path() {
    python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["models"][sys.argv[2]]["path"])' "$MANIFEST" "$1"
}

run_arm() {
    local arm=$1 model_key=$2 thinking=$3 mtp=$4
    local model; model=$(model_path "$model_key")
    local arm_dir="$OUT/$arm"; mkdir -p "$arm_dir"
    local -a server=("$SERVER" -m "$model" --host 127.0.0.1 --port "$PORT" --metrics --slots --jinja --reasoning on --reasoning-budget -1 --reasoning-format deepseek --device ROCm0 -ngl all -fa on -np 1 -c 49152 -t 8 -tb 8 -b 2048 -ub 2048 -ctk f16 -ctv f16)
    [[ "$mtp" == true ]] && server+=(--spec-type draft-mtp --spec-draft-n-max 1)
    printf '%q ' env GGML_IQK=1 taskset -c "$CORES" "${server[@]}" >"$arm_dir/server.argv"
    printf '\n' >>"$arm_dir/server.argv"
    env GGML_IQK=1 LD_LIBRARY_PATH=/mnt/raid0/llm/llama.cpp/build-hip/bin taskset -c "$CORES" "${server[@]}" >"$arm_dir/server.stdout" 2>"$arm_dir/server.stderr" &
    SERVER_PID=$!
    printf '{"pid":%s,"start_ticks":"%s"}\n' "$SERVER_PID" "$(awk '{print $22}' "/proc/$SERVER_PID/stat")" >"$arm_dir/server.launch.json"
    wait_for_health "$arm_dir"
    prove_listener "$arm_dir/listener.json"
    local suites=(swe_oracle lcb_hard)
    local suite n max question runner_suite pq summary
    for suite in "${suites[@]}"; do
        if [[ "$suite" == swe_oracle ]]; then
            n=40; max=3072; runner_suite=swebench_oracle
            question="$OUT/instrument/questions_swe_oracle.json"
        else
            n=53; max=4096; runner_suite=livecodebench_hard
            question="$OUT/instrument/questions_livecodebench_hard.json"
        fi
        pq="$arm_dir/$suite.sealed.jsonl"; summary="$arm_dir/$suite.summary.json"
        if [[ -f "$pq" && -f "$summary" && -f "${pq%.jsonl}.live-status.json" ]]; then
            validate_capture "$pq" "$summary" "$question" "$n" "$RUNNER_SHA" "$runner_suite" "$arm"
            if [[ "$suite" == swe_oracle ]]; then
                convert_swe_capture "$arm_dir" "$pq" "$arm"
            fi
            printf '27b-continuation: reused validator-clean capture %s/%s\n' "$arm" "$suite"
            continue
        fi
        local -a runner=(python3 -B "$OUT/instrument/v7_quality_gate_runner.py" --host 127.0.0.1 --port "$PORT" --output "$summary" --suites "$runner_suite" --n "$n" --limit "$n" --seed 42 --max-tokens "$max" --endpoint chat --kernel production-consolidated-v8 --concurrency 1 --repeats 1 --arm "$arm" --binary "$SERVER" --models "$model" --temperature 0.6 --top-p 0.95 --top-k 20 --questions-in "$question" --per-question-out "$pq")
        [[ "$thinking" == true ]] && runner+=(--enable-thinking) || runner+=(--no-enable-thinking)
        printf '%q ' env PYTHONPATH="$REPO/scripts/benchmark" RUNNER_REQUEST_TIMEOUT_S=3600 taskset -c "$CORES" "${runner[@]}" >"$arm_dir/$suite.evaluator.argv"
        printf '\n' >>"$arm_dir/$suite.evaluator.argv"
        local live_status="${pq%.jsonl}.live-status.json" runner_pid watchdog_pid
        env PYTHONPATH="$REPO/scripts/benchmark" RUNNER_REQUEST_TIMEOUT_S=3600 taskset -c "$CORES" "${runner[@]}" >"$arm_dir/$suite.evaluator.stdout" 2>"$arm_dir/$suite.evaluator.stderr" &
        runner_pid=$!
        wait_for_live_status "$live_status" "$runner_pid"
        python3 -B "$OUT/instrument/capture_integrity_watchdog.py" --watch \
            --poll-interval-s 5 --startup-grace-s 30 --stale-timeout-s 900 \
            --request-error-threshold 1 "$live_status" \
            >"$arm_dir/$suite.watchdog.stdout" 2>"$arm_dir/$suite.watchdog.stderr" &
        watchdog_pid=$!
        if ! await_runner_and_watchdog "$runner_pid" "$watchdog_pid"; then
            die "capture watchdog or runner failed for $arm/$suite"
        fi
        validate_capture "$pq" "$summary" "$question" "$n" "$RUNNER_SHA" "$runner_suite" "$arm"
        if [[ "$suite" == swe_oracle ]]; then
            convert_swe_capture "$arm_dir" "$pq" "$arm"
        fi
    done
    stop_owned_server
}

self_test() {
    local old_same=$SAME
    SAME=$(mktemp -d)
    if require_prerequisites; then die "self-test expected missing markers to fail"; fi
    rm -rf "$SAME"; SAME=$old_same

    # watchdog fails first: runner is terminated and watchdog status wins.
    sleep 60 &
    local fake_runner=$!
    (exit 7) &
    local fake_watchdog=$!
    local rc
    if await_runner_and_watchdog "$fake_runner" "$fake_watchdog"; then rc=0; else rc=$?; fi
    if (( rc == 0 )); then
        die "self-test expected watchdog failure to propagate"
    fi
    (( rc == 7 )) || die "self-test watchdog failure returned $rc, expected 7"
    ! ps -p "$fake_runner" >/dev/null 2>&1 || die "self-test runner survived watchdog failure"

    # watcher success first: let the runner finish naturally and return it.
    local natural_marker
    natural_marker=$(mktemp)
    rm -f "$natural_marker"
    bash -c 'sleep 0.2; touch "$1"; exit 9' -- "$natural_marker" &
    fake_runner=$!
    (exit 0) &
    fake_watchdog=$!
    if await_runner_and_watchdog "$fake_runner" "$fake_watchdog"; then rc=0; else rc=$?; fi
    if (( rc == 0 )); then
        die "self-test expected runner status after successful watcher"
    fi
    (( rc == 9 )) || die "self-test successful watcher returned $rc, expected 9"
    test -f "$natural_marker" || die "self-test watcher success killed healthy runner"
    ! ps -p "$fake_runner" >/dev/null 2>&1 || die "self-test natural runner remains live"
    rm -f "$natural_marker"

    # runner fails first: observer is stopped promptly and runner status wins.
    (exit 5) &
    fake_runner=$!
    sleep 60 &
    fake_watchdog=$!
    if await_runner_and_watchdog "$fake_runner" "$fake_watchdog"; then rc=0; else rc=$?; fi
    if (( rc == 0 )); then
        die "self-test expected runner failure to propagate"
    fi
    (( rc == 5 )) || die "self-test runner failure returned $rc, expected 5"
    ! ps -p "$fake_watchdog" >/dev/null 2>&1 || die "self-test watcher survived runner failure"

    # runner success first still requires later watchdog success.
    (exit 0) &
    fake_runner=$!
    bash -c 'sleep 0.2; exit 6' &
    fake_watchdog=$!
    if await_runner_and_watchdog "$fake_runner" "$fake_watchdog"; then rc=0; else rc=$?; fi
    if (( rc == 0 )); then
        die "self-test expected late watchdog failure to propagate"
    fi
    (( rc == 6 )) || die "self-test late watchdog failure returned $rc, expected 6"

    # Both sides complete successfully.
    bash -c 'sleep 0.1; exit 0' &
    fake_runner=$!
    bash -c 'sleep 0.2; exit 0' &
    fake_watchdog=$!
    await_runner_and_watchdog "$fake_runner" "$fake_watchdog"
    printf '27b-continuation self-test: PASS\n'
}

main() {
    case ${1:-} in
        --self-test) self_test ;;
        --execute)
            [[ $# -eq 1 ]] || die "usage: $0 --execute|--self-test"
            wait_for_prerequisites
            prepare_output_dir
            verify_frozen_inputs
            trap cleanup EXIT INT TERM
            run_arm A3-tc-quality__thinkingcap thinkingcap true false
            run_arm A3-ff-quality__stock_non_mtp stock_non_mtp false false
            run_arm A3-ff-quality__fable_non_mtp fable_non_mtp false false
            run_arm A3-ff-embedded-mtp__fable_mtp fable_mtp false true
            date -u +%Y-%m-%dT%H:%M:%SZ >"$OUT/continuation.complete"
            trap - EXIT
            ;;
        --resume-after-converter-fix)
            [[ $# -eq 1 ]] || die "usage: $0 --execute|--resume-after-converter-fix|--self-test"
            wait_for_prerequisites
            ! port_listening || die "port $PORT is occupied"
            repair_instrument_for_resume
            trap cleanup EXIT INT TERM
            run_arm A3-tc-quality__thinkingcap thinkingcap true false
            run_arm A3-ff-quality__stock_non_mtp stock_non_mtp false false
            run_arm A3-ff-quality__fable_non_mtp fable_non_mtp false false
            run_arm A3-ff-embedded-mtp__fable_mtp fable_mtp false true
            date -u +%Y-%m-%dT%H:%M:%SZ >"$OUT/continuation.complete"
            trap - EXIT
            ;;
        *) die "usage: $0 --execute|--resume-after-converter-fix|--self-test" ;;
    esac
}

main "$@"
